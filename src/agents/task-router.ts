/**
 * Task-based model router for intelligent model selection.
 *
 * Integrates task classification and model scoring to select the best model
 * based on task requirements. Works as a pre-processor before `runWithModelFallback`,
 * selecting the best primary model and optionally providing ranked fallbacks.
 *
 * @see {@link ./task-classifier.ts} for task classification
 * @see {@link ./model-scorer.ts} for model scoring
 * @see {@link ./model-fallback.ts} for fallback execution
 */

import type { OpenClawConfig } from "../config/config.js";
import type { ModelRoutingConfig, TaskRoutingRule } from "../config/types.agent-defaults.js";
import type { ModelScore, ModelScoringWeights } from "../config/types.models.js";
import type { ModelCatalogEntry } from "./model-catalog.js";
import type { ModelRef } from "./model-selection.js";
import type { TaskClassification, TaskClassificationHints } from "./task-classifier.js";
import { getChildLogger } from "../logging/logger.js";
import {
  ensureAuthProfileStore,
  isProfileInCooldown,
  resolveAuthProfileOrder,
} from "./auth-profiles.js";
import { loadModelCatalog } from "./model-catalog.js";
import {
  DEFAULT_SCORING_WEIGHTS,
  filterModelsByCapabilities,
  normalizeWeights,
  rankModels,
  explainScore,
} from "./model-scorer.js";
import { resolveDefaultModelForAgent } from "./model-selection.js";
import { classifyTask } from "./task-classifier.js";

const log = getChildLogger({ subsystem: "task-router" });

/** Default routing configuration. */
export const DEFAULT_ROUTING_CONFIG: Required<
  Omit<ModelRoutingConfig, "taskRules" | "scoringWeights">
> & {
  taskRules: TaskRoutingRule[];
  scoringWeights: ModelScoringWeights;
} = {
  enabled: false,
  strategy: "balanced",
  preferLocal: true,
  localProviders: ["ollama"],
  cloudProviders: ["anthropic", "openai", "google"],
  taskRules: [],
  scoringWeights: DEFAULT_SCORING_WEIGHTS,
  fallbackBehavior: "default-model",
};

/** Routing decision metadata for debugging and logging. */
export type RoutingDecision = {
  /** Score of the selected model. */
  selectedScore: ModelScore;
  /** Task classification used for routing. */
  taskClassification: TaskClassification;
  /** All ranked model scores. */
  allRankedScores: ModelScore[];
  /** Human-readable reasoning for the decision. */
  reasoning: string;
};

/** Result of model routing. */
export type RoutingResult = {
  /** Primary model selected. */
  primary: ModelRef;
  /** Optional fallback models in priority order. */
  fallbacks?: ModelRef[];
  /** Routing decision metadata. */
  decision: RoutingDecision;
};

/** Result when manual selection is required. */
export type ManualSelectionRequiredResult = {
  manualSelectionRequired: true;
  provider?: undefined;
  model?: undefined;
  fallbacks?: undefined;
  decision?: undefined;
};

/** Result of routeModelForTask. */
export type RouteModelForTaskResult =
  | {
      provider: string;
      model: string;
      fallbacks?: string[];
      decision?: RoutingDecision;
      manualSelectionRequired?: false;
    }
  | ManualSelectionRequiredResult;

/** Options for task-based model selection. */
export type TaskRoutingOptions = {
  /** User prompt text. */
  prompt: string;
  /** Whether the request includes images. */
  hasImages: boolean;
  /** Optional task classification hints. */
  hints?: TaskClassificationHints;
  /** Manual model override (bypasses routing). */
  overrideModel?: string;
  /** Allowed providers filter. */
  allowedProviders?: string[];
};

/**
 * Task-based model router that integrates task classification and model scoring.
 *
 * @example
 * ```typescript
 * const router = new TaskBasedModelRouter(routingConfig, catalog);
 * const result = router.selectModelForTask({
 *   prompt: "Fix the bug in auth.ts",
 *   hasImages: false,
 * });
 * console.log(`Selected: ${result.primary.provider}/${result.primary.model}`);
 * ```
 */
export class TaskBasedModelRouter {
  private readonly config: Required<Omit<ModelRoutingConfig, "taskRules" | "scoringWeights">> & {
    taskRules: TaskRoutingRule[];
    scoringWeights: ModelScoringWeights;
  };
  private readonly catalog: ModelCatalogEntry[];

  /**
   * Create a new task-based model router.
   *
   * @param config - Routing configuration
   * @param catalog - Available model catalog entries
   */
  constructor(config: ModelRoutingConfig, catalog: ModelCatalogEntry[]) {
    this.config = mergeRoutingConfig(config);
    this.catalog = catalog;
  }

  /**
   * Select the best model for a given task.
   *
   * @param options - Task routing options
   * @returns Routing result with primary model, fallbacks, and decision metadata
   *
   * @example
   * ```typescript
   * const result = router.selectModelForTask({
   *   prompt: "Debug this TypeScript function",
   *   hasImages: false,
   * });
   * console.log(result.decision.reasoning);
   * ```
   */
  selectModelForTask(options: TaskRoutingOptions): RoutingResult | null {
    const { prompt, hasImages, hints, overrideModel, allowedProviders } = options;

    // Handle manual model override
    if (overrideModel?.trim()) {
      return this.handleModelOverride(overrideModel, prompt, hasImages, hints);
    }

    // Check if routing is enabled
    if (!this.config.enabled) {
      log.debug("Task-based routing is disabled");
      return null;
    }

    try {
      // Classify the task
      const taskClassification = classifyTask(prompt, hasImages, hints);

      // Filter catalog by allowed providers if specified
      let filteredCatalog = this.catalog;
      if (allowedProviders && allowedProviders.length > 0) {
        filteredCatalog = filterModelsByCapabilities(this.catalog, {
          allowedProviders,
        });
        if (filteredCatalog.length === 0) {
          log.warn("No models match allowedProviders filter, using full catalog", {
            allowedProviders,
          });
          filteredCatalog = this.catalog;
        }
      }

      // Apply task rules to filter and boost models
      filteredCatalog = this.applyTaskRules(filteredCatalog, taskClassification);

      if (filteredCatalog.length === 0) {
        log.warn(
          "No suitable models found after applying task rules, falling back to full catalog",
        );
        filteredCatalog = this.catalog;
      }

      // Apply routing strategy to adjust weights
      const adjustedWeights = this.applyRoutingStrategy(this.config.scoringWeights);

      // Rank models
      const rankedScores = rankModels(filteredCatalog, taskClassification, adjustedWeights);

      if (rankedScores.length === 0) {
        log.warn("No models could be ranked, routing failed");
        return null;
      }

      // Apply provider preference boost based on task complexity
      const { scores: finalRankedScores, preference } = this.applyProviderPreference(
        rankedScores,
        taskClassification,
      );

      const selectedScore = finalRankedScores[0];
      const primary: ModelRef = {
        provider: selectedScore.provider,
        model: selectedScore.model,
      };

      // Generate fallback list
      const fallbacks = this.generateFallbackList(finalRankedScores, primary);

      // Build reasoning string
      const baseReasoning = explainScore(selectedScore, taskClassification.type);
      const reasoning = `Task-based routing: ${baseReasoning} | Strategy: ${this.config.strategy} | Provider preference: ${preference}`;

      const decision: RoutingDecision = {
        selectedScore,
        taskClassification,
        allRankedScores: finalRankedScores,
        reasoning,
      };

      // Log routing decision
      log.info("Model selected via task-based routing", {
        taskType: taskClassification.type,
        complexity: taskClassification.complexity,
        selectedProvider: primary.provider,
        selectedModel: primary.model,
        totalScore: selectedScore.totalScore,
        strategy: this.config.strategy,
        preferLocal: this.config.preferLocal,
        providerPreference: preference,
      });

      log.debug("Routing decision details", {
        allRankedModels: finalRankedScores.map((s) => ({
          provider: s.provider,
          model: s.model,
          score: s.totalScore,
        })),
        fallbackCount: fallbacks?.length ?? 0,
      });

      return { primary, fallbacks, decision };
    } catch (err) {
      log.error("Routing error, falling back to default model selection", {
        error: err instanceof Error ? err.message : String(err),
        stack: err instanceof Error ? err.stack : undefined,
      });
      return null;
    }
  }

  /**
   * Handle manual model override.
   */
  private handleModelOverride(
    overrideModel: string,
    prompt: string,
    hasImages: boolean,
    hints?: TaskClassificationHints,
  ): RoutingResult {
    return buildManualOverrideRoutingResult({ overrideModel, prompt, hasImages, hints });
  }

  /**
   * Apply routing strategy to adjust scoring weights.
   */
  private applyRoutingStrategy(baseWeights: ModelScoringWeights): ModelScoringWeights {
    const strategy = this.config.strategy;

    switch (strategy) {
      case "cost-optimized":
        return normalizeWeights({
          capabilityMatch: baseWeights.capabilityMatch * 0.8,
          costEfficiency: 0.5,
          performance: 0.1,
          availability: baseWeights.availability,
        });

      case "performance-optimized":
        return normalizeWeights({
          capabilityMatch: baseWeights.capabilityMatch * 0.8,
          costEfficiency: 0.1,
          performance: 0.5,
          availability: baseWeights.availability,
        });

      case "balanced":
      default:
        return normalizeWeights(baseWeights);
    }
  }

  /**
   * Apply task rules to filter and boost models.
   */
  private applyTaskRules(
    catalog: ModelCatalogEntry[],
    task: TaskClassification,
  ): ModelCatalogEntry[] {
    const rules = this.config.taskRules;
    if (rules.length === 0) {
      return catalog;
    }

    // Find matching rules for this task
    const matchingRules = rules.filter((rule) => {
      if (rule.taskType !== task.type) {
        return false;
      }
      if (rule.complexity && rule.complexity !== task.complexity) {
        return false;
      }
      return true;
    });

    if (matchingRules.length === 0) {
      return catalog;
    }

    let filtered = catalog;

    for (const rule of matchingRules) {
      // Apply excludeProviders filter
      if (rule.excludeProviders && rule.excludeProviders.length > 0) {
        const excludeSet = new Set(rule.excludeProviders.map((p) => p.toLowerCase()));
        filtered = filtered.filter((entry) => !excludeSet.has(entry.provider.toLowerCase()));
      }

      // Apply preferredProviders filter (if specified, only include these)
      if (rule.preferredProviders && rule.preferredProviders.length > 0) {
        const preferredSet = new Set(rule.preferredProviders.map((p) => p.toLowerCase()));
        const preferredModels = filtered.filter((entry) =>
          preferredSet.has(entry.provider.toLowerCase()),
        );
        // Only apply if we have matches, otherwise keep all
        if (preferredModels.length > 0) {
          filtered = preferredModels;
        }
      }

      // Apply preferredModels filter
      if (rule.preferredModels && rule.preferredModels.length > 0) {
        const preferredModelSet = new Set(rule.preferredModels.map((m) => m.toLowerCase()));
        const preferredModels = filtered.filter((entry) => {
          const key = `${entry.provider}/${entry.id}`.toLowerCase();
          return preferredModelSet.has(key);
        });
        if (preferredModels.length > 0) {
          filtered = preferredModels;
        }
      }

      // Apply hard requirements via filterModelsByCapabilities
      filtered = filterModelsByCapabilities(filtered, {
        requiresVision: task.requiresVision,
        requiresReasoning: rule.requireReasoning ?? task.requiresReasoning,
        minContextWindow: rule.minContextWindow,
      });
    }

    return filtered;
  }

  /**
   * Apply provider preference based on task complexity and reasoning needs.
   * - Simple tasks prefer local providers (if preferLocal is enabled).
   * - Complex or reasoning tasks prefer cloud providers.
   */
  private applyProviderPreference(
    rankedScores: ModelScore[],
    taskClassification: TaskClassification,
  ): { scores: ModelScore[]; preference: "local" | "cloud" | "none" } {
    const localProviderSet = new Set(this.config.localProviders.map((p) => p.toLowerCase()));
    const cloudProviderSet = new Set(this.config.cloudProviders.map((p) => p.toLowerCase()));

    const prefersCloud =
      taskClassification.complexity === "complex" || taskClassification.requiresReasoning;
    const prefersLocal =
      this.config.preferLocal && taskClassification.complexity === "simple" && !prefersCloud;

    let preference: "local" | "cloud" | "none" = "none";
    let preferredProviders: Set<string> | null = null;

    if (prefersCloud && cloudProviderSet.size > 0) {
      preference = "cloud";
      preferredProviders = cloudProviderSet;
    } else if (prefersLocal && localProviderSet.size > 0) {
      preference = "local";
      preferredProviders = localProviderSet;
    }

    if (!preferredProviders) {
      return { scores: rankedScores, preference };
    }

    // Boost preferred provider scores by 10%
    const boostedScores = rankedScores.map((score) => {
      const isPreferred = preferredProviders?.has(score.provider.toLowerCase());
      if (isPreferred) {
        return {
          ...score,
          totalScore: Math.min(score.totalScore * 1.1, 1.0),
        };
      }
      return score;
    });

    // Re-sort after boosting
    return { scores: boostedScores.toSorted((a, b) => b.totalScore - a.totalScore), preference };
  }

  /**
   * Generate fallback list from ranked scores.
   */
  private generateFallbackList(
    rankedScores: ModelScore[],
    primary: ModelRef,
    maxFallbacks = 3,
  ): ModelRef[] | undefined {
    const fallbacks: ModelRef[] = [];
    const primaryKey = `${primary.provider}/${primary.model}`.toLowerCase();

    for (const score of rankedScores) {
      if (fallbacks.length >= maxFallbacks) {
        break;
      }
      const key = `${score.provider}/${score.model}`.toLowerCase();
      if (key === primaryKey) {
        continue;
      }
      fallbacks.push({
        provider: score.provider,
        model: score.model,
      });
    }

    return fallbacks.length > 0 ? fallbacks : undefined;
  }
}

/**
 * Merge user routing config with defaults.
 *
 * @param config - User-provided routing configuration
 * @returns Complete routing configuration with defaults applied
 */
export function mergeRoutingConfig(config: ModelRoutingConfig | undefined): Required<
  Omit<ModelRoutingConfig, "taskRules" | "scoringWeights">
> & {
  taskRules: TaskRoutingRule[];
  scoringWeights: ModelScoringWeights;
} {
  if (!config) {
    return { ...DEFAULT_ROUTING_CONFIG };
  }

  return {
    enabled: config.enabled ?? DEFAULT_ROUTING_CONFIG.enabled,
    strategy: config.strategy ?? DEFAULT_ROUTING_CONFIG.strategy,
    preferLocal: config.preferLocal ?? DEFAULT_ROUTING_CONFIG.preferLocal,
    localProviders: config.localProviders ?? DEFAULT_ROUTING_CONFIG.localProviders,
    cloudProviders: config.cloudProviders ?? DEFAULT_ROUTING_CONFIG.cloudProviders,
    taskRules: config.taskRules ?? DEFAULT_ROUTING_CONFIG.taskRules,
    scoringWeights: config.scoringWeights
      ? normalizeWeights(config.scoringWeights)
      : DEFAULT_ROUTING_CONFIG.scoringWeights,
    fallbackBehavior: config.fallbackBehavior ?? DEFAULT_ROUTING_CONFIG.fallbackBehavior,
  };
}

/**
 * Load routing configuration from OpenClawConfig.
 *
 * @param cfg - OpenClaw configuration
 * @returns Routing configuration or default if not configured
 */
export function loadRoutingConfig(cfg: OpenClawConfig | undefined): ModelRoutingConfig {
  if (!cfg?.agents?.defaults?.routing) {
    return { enabled: false };
  }
  return cfg.agents.defaults.routing;
}

/**
 * Check if routing should be applied.
 *
 * @param params - Check parameters
 * @returns True if routing should be applied
 *
 * @example
 * ```typescript
 * if (shouldUseRouting({ cfg, overrideModel })) {
 *   const result = router.selectModelForTask({ prompt, hasImages });
 * }
 * ```
 */
export function shouldUseRouting(params: {
  cfg: OpenClawConfig | undefined;
  overrideModel?: string;
}): boolean {
  // Manual override bypasses routing check
  if (params.overrideModel?.trim()) {
    return true;
  }

  const routingConfig = loadRoutingConfig(params.cfg);
  return routingConfig.enabled === true;
}

/**
 * Create a router from configuration.
 *
 * @param params - Creation parameters
 * @returns TaskBasedModelRouter instance or null if routing is disabled
 *
 * @example
 * ```typescript
 * const router = createRouterFromConfig({ cfg, catalog });
 * if (router) {
 *   const result = router.selectModelForTask({ prompt, hasImages });
 * }
 * ```
 */
export function createRouterFromConfig(params: {
  cfg: OpenClawConfig | undefined;
  catalog: ModelCatalogEntry[];
}): TaskBasedModelRouter | null {
  const routingConfig = loadRoutingConfig(params.cfg);

  if (!routingConfig.enabled) {
    return null;
  }

  return new TaskBasedModelRouter(routingConfig, params.catalog);
}

/**
 * High-level function to route model selection for a task.
 *
 * Combines catalog loading, routing, and fallback handling into a single call.
 *
 * @param params - Routing parameters
 * @returns Selected model with optional fallbacks and decision metadata, or a manual-selection sentinel
 *
 * @example
 * ```typescript
 * const result = await routeModelForTask({
 *   cfg,
 *   prompt: "Fix the bug in auth.ts",
 *   hasImages: false,
 * });
 * console.log(`Using: ${result.provider}/${result.model}`);
 * ```
 */
export async function routeModelForTask(params: {
  cfg: OpenClawConfig | undefined;
  prompt: string;
  hasImages: boolean;
  hints?: TaskClassificationHints;
  overrideModel?: string;
  allowedProviders?: string[];
  agentDir?: string;
}): Promise<RouteModelForTaskResult> {
  const { cfg, prompt, hasImages, hints, overrideModel, allowedProviders, agentDir } = params;

  // Manual override bypasses routing configuration entirely
  if (overrideModel?.trim()) {
    const overrideResult = buildManualOverrideRoutingResult({
      overrideModel,
      prompt,
      hasImages,
      hints,
    });

    return {
      provider: overrideResult.primary.provider,
      model: overrideResult.primary.model,
      decision: overrideResult.decision,
    };
  }

  // Load catalog
  const catalog = await loadModelCatalog({ config: cfg });

  // Create router
  const router = createRouterFromConfig({ cfg, catalog });

  if (router) {
    const result = router.selectModelForTask({
      prompt,
      hasImages,
      hints,
      overrideModel,
      allowedProviders,
    });

    if (result) {
      // Filter fallbacks by auth profile cooldown if agentDir is provided
      let filteredFallbacks = result.fallbacks;
      if (filteredFallbacks && agentDir && cfg) {
        filteredFallbacks = filterFallbacksByCooldown({
          cfg,
          agentDir,
          fallbacks: filteredFallbacks,
        });
      }

      return {
        provider: result.primary.provider,
        model: result.primary.model,
        fallbacks: filteredFallbacks?.map((f) => `${f.provider}/${f.model}`),
        decision: result.decision,
      };
    }
  }

  // Routing disabled or failed, fall back to default model
  const routingConfig = loadRoutingConfig(cfg);
  const fallbackBehavior = routingConfig.fallbackBehavior ?? "default-model";

  if (fallbackBehavior === "manual-selection") {
    log.warn("Routing fallback requires manual selection", {
      fallbackBehavior,
    });
    return {
      manualSelectionRequired: true,
    };
  }

  // default-model behavior
  const defaultRef = resolveDefaultModelForAgent({ cfg: cfg ?? {} });
  log.debug("Using default model (routing disabled or failed)", {
    provider: defaultRef.provider,
    model: defaultRef.model,
  });

  return {
    provider: defaultRef.provider,
    model: defaultRef.model,
  };
}

function parseOverrideModelRef(overrideModel: string): ModelRef {
  const trimmed = overrideModel.trim();
  const slashIndex = trimmed.indexOf("/");

  if (slashIndex === -1) {
    // Assume default provider (anthropic)
    return { provider: "anthropic", model: trimmed };
  }

  return {
    provider: trimmed.slice(0, slashIndex).trim(),
    model: trimmed.slice(slashIndex + 1).trim(),
  };
}

function buildManualOverrideRoutingResult(params: {
  overrideModel: string;
  prompt: string;
  hasImages: boolean;
  hints?: TaskClassificationHints;
}): RoutingResult {
  const { overrideModel, prompt, hasImages, hints } = params;
  const primary = parseOverrideModelRef(overrideModel);

  log.info("Manual model override", {
    provider: primary.provider,
    model: primary.model,
  });

  const taskClassification = classifyTask(prompt, hasImages, hints);

  const decision: RoutingDecision = {
    selectedScore: {
      provider: primary.provider,
      model: primary.model,
      totalScore: 1.0,
      breakdown: { capability: 1.0, cost: 1.0, performance: 1.0, availability: 1.0 },
      reasoning: `Manual override: ${primary.provider}/${primary.model}`,
    },
    taskClassification,
    allRankedScores: [],
    reasoning: `Manual model override: ${primary.provider}/${primary.model}`,
  };

  return {
    primary,
    decision,
  };
}

/**
 * Filter fallbacks by auth profile cooldown status.
 */
function filterFallbacksByCooldown(params: {
  cfg: OpenClawConfig;
  agentDir: string;
  fallbacks: ModelRef[];
}): ModelRef[] {
  const { cfg, agentDir, fallbacks } = params;

  try {
    const authStore = ensureAuthProfileStore(agentDir, { allowKeychainPrompt: false });

    return fallbacks.filter((fallback) => {
      const profileIds = resolveAuthProfileOrder({
        cfg,
        store: authStore,
        provider: fallback.provider,
      });

      // If no profiles configured, allow the fallback
      if (profileIds.length === 0) {
        return true;
      }

      // Check if at least one profile is available
      return profileIds.some((id) => !isProfileInCooldown(authStore, id));
    });
  } catch {
    // On error, return all fallbacks
    return fallbacks;
  }
}

/**
 * Generate fallback list as string array from routing result.
 *
 * @param rankedScores - Ranked model scores
 * @param primary - Primary model to exclude
 * @param maxFallbacks - Maximum number of fallbacks (default: 3)
 * @returns Fallback model refs in "provider/model" format
 */
export function generateFallbackList(
  rankedScores: ModelScore[],
  primary: ModelRef,
  maxFallbacks = 3,
): string[] {
  const fallbacks: string[] = [];
  const primaryKey = `${primary.provider}/${primary.model}`.toLowerCase();

  for (const score of rankedScores) {
    if (fallbacks.length >= maxFallbacks) {
      break;
    }
    const key = `${score.provider}/${score.model}`.toLowerCase();
    if (key === primaryKey) {
      continue;
    }
    fallbacks.push(`${score.provider}/${score.model}`);
  }

  return fallbacks;
}
