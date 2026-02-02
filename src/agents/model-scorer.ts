/**
 * Model scoring engine for intelligent task-based model routing.
 *
 * Ranks available models based on task requirements using the existing
 * task classification and capability inference. Evaluates models across
 * four dimensions: capability match, cost efficiency, performance, and availability.
 *
 * @see {@link ../config/types.models.ts} for scoring types
 * @see {@link ./task-classifier.ts} for task classification
 * @see {@link ./model-capabilities.ts} for capability inference
 */

import type {
  ModelCapabilities,
  ModelCostTier,
  ModelScore,
  ModelScoreBreakdown,
  ModelScoringWeights,
  ModelTaskComplexity,
} from "../config/types.models.js";
import type { ModelCatalogEntry } from "./model-catalog.js";
import type { TaskClassification, TaskClassificationHints } from "./task-classifier.js";
import { modelCatalogEntryToCapabilities } from "./model-catalog.js";
import { classifyTask } from "./task-classifier.js";

/** Default weights for model scoring. */
export const DEFAULT_SCORING_WEIGHTS: ModelScoringWeights = {
  capabilityMatch: 0.4,
  costEfficiency: 0.3,
  performance: 0.2,
  availability: 0.1,
};

/** Cost tier score mapping. */
const COST_TIER_SCORES: Record<ModelCostTier, number> = {
  free: 1.0,
  low: 0.7,
  medium: 0.4,
  high: 0.2,
};

/** Complexity level numeric values for comparison. */
const COMPLEXITY_LEVELS: Record<ModelTaskComplexity, number> = {
  simple: 1,
  moderate: 2,
  complex: 3,
};

/**
 * Normalize weights to sum to 1.0.
 *
 * @param weights - Weights to normalize
 * @returns Normalized weights
 */
export function normalizeWeights(weights: ModelScoringWeights): ModelScoringWeights {
  const sum =
    weights.capabilityMatch + weights.costEfficiency + weights.performance + weights.availability;
  if (sum === 0) {
    return DEFAULT_SCORING_WEIGHTS;
  }
  if (Math.abs(sum - 1.0) < 0.001) {
    return weights;
  }
  return {
    capabilityMatch: weights.capabilityMatch / sum,
    costEfficiency: weights.costEfficiency / sum,
    performance: weights.performance / sum,
    availability: weights.availability / sum,
  };
}

/**
 * Get capabilities for a model, inferring if not present.
 */
function getModelCapabilities(model: ModelCatalogEntry): ModelCapabilities {
  if (model.capabilities) {
    return model.capabilities;
  }
  return modelCatalogEntryToCapabilities(model);
}

/**
 * Score capability match between a model and task requirements.
 *
 * Evaluates:
 * - Task type match: model's taskTypes includes the detected task type
 * - Complexity support: model's maxComplexity >= task complexity
 * - Vision requirement: if task requires vision, model must support it
 * - Reasoning requirement: if task requires reasoning, model should support it
 * - Context window: penalize if estimatedContextTokens > model's contextWindow
 *
 * @param model - Model to score
 * @param task - Task classification
 * @returns Score from 0-1
 */
export function scoreCapabilityMatch(model: ModelCatalogEntry, task: TaskClassification): number {
  const capabilities = getModelCapabilities(model);
  let score = 0;
  let factors = 0;

  // Task type match (weight: 0.3)
  const taskTypeMatch = capabilities.taskTypes.includes(task.type);
  score += taskTypeMatch ? 0.3 : 0;
  factors += 0.3;

  // Complexity support (weight: 0.25)
  const modelComplexity = COMPLEXITY_LEVELS[capabilities.maxComplexity];
  const taskComplexity = COMPLEXITY_LEVELS[task.complexity];
  const complexityMatch = modelComplexity >= taskComplexity;
  score += complexityMatch ? 0.25 : 0.25 * (modelComplexity / taskComplexity);
  factors += 0.25;

  // Vision requirement (weight: 0.2)
  if (task.requiresVision) {
    score += capabilities.supportsVision ? 0.2 : 0;
    factors += 0.2;
  } else {
    // No vision required, give full points
    score += 0.2;
    factors += 0.2;
  }

  // Reasoning requirement (weight: 0.15)
  if (task.requiresReasoning) {
    score += capabilities.supportsReasoning ? 0.15 : 0.05;
    factors += 0.15;
  } else {
    score += 0.15;
    factors += 0.15;
  }

  // Context window (weight: 0.1)
  if (task.estimatedContextTokens !== undefined && task.estimatedContextTokens > 0) {
    const contextRatio = capabilities.contextWindow / task.estimatedContextTokens;
    if (contextRatio >= 1) {
      score += 0.1;
    } else {
      score += 0.1 * contextRatio;
    }
  } else {
    score += 0.1;
  }
  factors += 0.1;

  return factors > 0 ? score / factors : 0;
}

/**
 * Score cost efficiency of a model.
 *
 * @param model - Model to score
 * @returns Score from 0-1 (1.0 = free, 0.2 = high cost)
 */
export function scoreCostEfficiency(model: ModelCatalogEntry): number {
  const capabilities = getModelCapabilities(model);
  return COST_TIER_SCORES[capabilities.costTier] ?? 0.4;
}

/**
 * Score performance characteristics of a model for a task.
 *
 * Evaluates:
 * - Context window size (larger is better for complex tasks)
 * - Reasoning capability bonus for reasoning/complex tasks
 *
 * @param model - Model to score
 * @param task - Task classification
 * @returns Score from 0-1
 */
export function scorePerformance(model: ModelCatalogEntry, task: TaskClassification): number {
  const capabilities = getModelCapabilities(model);
  let score = 0;

  // Context window scoring (0-0.6)
  // Normalize context window: 8k = 0.2, 32k = 0.4, 128k = 0.5, 200k+ = 0.6
  const contextWindow = capabilities.contextWindow;
  if (contextWindow >= 200_000) {
    score += 0.6;
  } else if (contextWindow >= 128_000) {
    score += 0.5;
  } else if (contextWindow >= 32_000) {
    score += 0.4;
  } else if (contextWindow >= 8_000) {
    score += 0.2;
  } else {
    score += 0.1;
  }

  // Reasoning bonus for complex/reasoning tasks (0-0.4)
  if (task.requiresReasoning || task.complexity === "complex") {
    score += capabilities.supportsReasoning ? 0.4 : 0.1;
  } else {
    score += 0.3;
  }

  return Math.min(score, 1.0);
}

/**
 * Score availability of a model.
 *
 * Currently returns 1.0 by default. Can be extended to integrate with:
 * - Provider health checks
 * - Auth profile cooldown status (isProfileInCooldown)
 *
 * @param _model - Model to score
 * @param providerHealth - Optional provider health status (0-1)
 * @returns Score from 0-1
 */
export function scoreAvailability(_model: ModelCatalogEntry, providerHealth?: number): number {
  if (typeof providerHealth === "number" && Number.isFinite(providerHealth)) {
    return Math.max(0, Math.min(1, providerHealth));
  }
  return 1.0;
}

/**
 * Score a model for a given task.
 *
 * Combines capability match, cost efficiency, performance, and availability
 * scores using configurable weights.
 *
 * @param params - Scoring parameters
 * @param params.model - Model to score
 * @param params.taskClassification - Task classification result
 * @param params.weights - Optional custom weights (defaults: capability 0.4, cost 0.3, performance 0.2, availability 0.1)
 * @param params.providerHealth - Optional provider health status (0-1)
 * @returns Complete model score with breakdown
 *
 * @example
 * ```typescript
 * const task = classifyTask("Fix the bug in auth.ts", false);
 * const score = scoreModel({ model: ollamaModel, taskClassification: task });
 * console.log(score.totalScore); // 0.85
 * ```
 */
export function scoreModel(params: {
  model: ModelCatalogEntry;
  taskClassification: TaskClassification;
  weights?: ModelScoringWeights;
  providerHealth?: number;
}): ModelScore {
  const { model, taskClassification, providerHealth } = params;
  const weights = normalizeWeights(params.weights ?? DEFAULT_SCORING_WEIGHTS);

  const breakdown: ModelScoreBreakdown = {
    capability: scoreCapabilityMatch(model, taskClassification),
    cost: scoreCostEfficiency(model),
    performance: scorePerformance(model, taskClassification),
    availability: scoreAvailability(model, providerHealth),
  };

  const totalScore =
    breakdown.capability * weights.capabilityMatch +
    breakdown.cost * weights.costEfficiency +
    breakdown.performance * weights.performance +
    breakdown.availability * weights.availability;

  return {
    provider: model.provider,
    model: model.id,
    totalScore,
    breakdown,
    reasoning: explainScore(
      {
        provider: model.provider,
        model: model.id,
        totalScore,
        breakdown,
        reasoning: "",
      },
      taskClassification.type,
    ),
  };
}

/**
 * Generate human-readable explanation for a model score.
 *
 * @param score - Model score to explain
 * @param taskType - Task type for context
 * @returns Human-readable reasoning string
 */
export function explainScore(score: ModelScore, taskType: string): string {
  const capPct = Math.round(score.breakdown.capability * 100);
  const costPct = Math.round(score.breakdown.cost * 100);
  const perfPct = Math.round(score.breakdown.performance * 100);
  const availPct = Math.round(score.breakdown.availability * 100);

  return `Selected ${score.provider}/${score.model} for ${taskType} task: ${capPct}% capability match, ${costPct}% cost efficiency, ${perfPct}% performance, ${availPct}% availability`;
}

/**
 * Rank all models by score for a given task.
 *
 * @param models - Model catalog entries to rank
 * @param task - Task classification
 * @param weights - Optional custom weights
 * @returns Array of ModelScore objects sorted by totalScore descending
 *
 * @example
 * ```typescript
 * const catalog = await loadModelCatalog();
 * const task = classifyTask("Analyze this image", true);
 * const ranked = rankModels(catalog, task);
 * const bestModel = ranked[0];
 * ```
 */
export function rankModels(
  models: ModelCatalogEntry[],
  task: TaskClassification,
  weights?: ModelScoringWeights,
): ModelScore[] {
  if (models.length === 0) {
    console.warn("[model-scorer] Empty model catalog provided");
    return [];
  }

  const scores = models.map((model) => scoreModel({ model, taskClassification: task, weights }));

  // Sort by total score descending, then by cost tier (free first), then by context window
  return scores.toSorted((a, b) => {
    if (Math.abs(a.totalScore - b.totalScore) > 0.001) {
      return b.totalScore - a.totalScore;
    }
    // Tie-breaker: prefer lower cost
    if (a.breakdown.cost !== b.breakdown.cost) {
      return b.breakdown.cost - a.breakdown.cost;
    }
    // Tie-breaker: prefer larger context window (higher performance score)
    return b.breakdown.performance - a.breakdown.performance;
  });
}

/** Requirements for filtering models. */
export type ModelFilterRequirements = {
  /** Require vision support. */
  requiresVision?: boolean;
  /** Require reasoning support. */
  requiresReasoning?: boolean;
  /** Minimum context window size. */
  minContextWindow?: number;
  /** Allowed providers (if specified, only these providers are included). */
  allowedProviders?: string[];
};

/**
 * Pre-filter models by hard requirements before scoring.
 *
 * @param models - Models to filter
 * @param requirements - Filter requirements
 * @returns Filtered models that meet all requirements
 */
export function filterModelsByCapabilities(
  models: ModelCatalogEntry[],
  requirements: ModelFilterRequirements,
): ModelCatalogEntry[] {
  return models.filter((model) => {
    const capabilities = getModelCapabilities(model);

    if (requirements.requiresVision && !capabilities.supportsVision) {
      return false;
    }

    if (requirements.requiresReasoning && !capabilities.supportsReasoning) {
      return false;
    }

    if (
      requirements.minContextWindow !== undefined &&
      capabilities.contextWindow < requirements.minContextWindow
    ) {
      return false;
    }

    if (requirements.allowedProviders && requirements.allowedProviders.length > 0) {
      const normalizedProvider = model.provider.toLowerCase();
      const allowed = requirements.allowedProviders.some(
        (p) => p.toLowerCase() === normalizedProvider,
      );
      if (!allowed) {
        return false;
      }
    }

    return true;
  });
}

/** Result of selecting the best model for a task. */
export type BestModelResult = {
  /** Top-ranked model score. */
  score: ModelScore;
  /** Task classification used. */
  task: TaskClassification;
  /** All ranked models (for fallback). */
  allRanked: ModelScore[];
};

/**
 * High-level function to select the best model for a task.
 *
 * Combines task classification and model ranking into a single call.
 *
 * @param params - Selection parameters
 * @param params.catalog - Available models
 * @param params.prompt - User prompt text
 * @param params.hasImages - Whether the request includes images
 * @param params.hints - Optional task classification hints
 * @param params.weights - Optional scoring weights
 * @returns Best model result with score, task classification, and all ranked models
 *
 * @example
 * ```typescript
 * const catalog = await loadModelCatalog();
 * const result = selectBestModelForTask({
 *   catalog,
 *   prompt: "Debug this TypeScript function",
 *   hasImages: false,
 * });
 * console.log(result.score.reasoning);
 * ```
 */
export function selectBestModelForTask(params: {
  catalog: ModelCatalogEntry[];
  prompt: string;
  hasImages: boolean;
  hints?: TaskClassificationHints;
  weights?: ModelScoringWeights;
}): BestModelResult | null {
  const { catalog, prompt, hasImages, hints, weights } = params;

  if (catalog.length === 0) {
    console.warn("[model-scorer] Empty model catalog, cannot select model");
    return null;
  }

  const task = classifyTask(prompt, hasImages, hints);

  // Pre-filter by hard requirements
  const requirements: ModelFilterRequirements = {
    requiresVision: task.requiresVision,
    requiresReasoning: task.requiresReasoning,
  };
  const filtered = filterModelsByCapabilities(catalog, requirements);

  // If no models match hard requirements, fall back to all models
  const modelsToRank = filtered.length > 0 ? filtered : catalog;

  const allRanked = rankModels(modelsToRank, task, weights);

  if (allRanked.length === 0) {
    return null;
  }

  return {
    score: allRanked[0],
    task,
    allRanked,
  };
}
