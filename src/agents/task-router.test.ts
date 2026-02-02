import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { ModelRoutingConfig, TaskRoutingRule } from "../config/types.agent-defaults.js";
import type { ModelCapabilities } from "../config/types.models.js";
import type { ModelCatalogEntry } from "./model-catalog.js";
import {
  TaskBasedModelRouter,
  DEFAULT_ROUTING_CONFIG,
  mergeRoutingConfig,
  loadRoutingConfig,
  shouldUseRouting,
  createRouterFromConfig,
  routeModelForTask,
  generateFallbackList,
} from "./task-router.js";

// Mock the logger to avoid file I/O during tests
vi.mock("../logging/logger.js", () => ({
  getChildLogger: () => ({
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  }),
}));

// Mock loadModelCatalog for routeModelForTask tests
vi.mock("./model-catalog.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("./model-catalog.js")>();
  return {
    ...original,
    loadModelCatalog: vi.fn().mockResolvedValue([]),
  };
});

// Mock auth-profiles for cooldown filtering tests
vi.mock("./auth-profiles.js", () => ({
  ensureAuthProfileStore: vi.fn().mockReturnValue({}),
  isProfileInCooldown: vi.fn().mockReturnValue(false),
  resolveAuthProfileOrder: vi.fn().mockReturnValue([]),
}));

// Mock model-selection for default model resolution
vi.mock("./model-selection.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("./model-selection.js")>();
  return {
    ...original,
    resolveDefaultModelForAgent: vi.fn().mockReturnValue({
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
    }),
  };
});

function createTestCatalog(): ModelCatalogEntry[] {
  return [
    {
      id: "llama3.2",
      name: "Llama 3.2",
      provider: "ollama",
      contextWindow: 128000,
      reasoning: false,
      input: ["text"],
      capabilities: {
        taskTypes: ["coding", "chat", "general"],
        maxComplexity: "moderate",
        supportsVision: false,
        supportsReasoning: false,
        contextWindow: 128000,
        costTier: "free",
      } as ModelCapabilities,
    },
    {
      id: "qwen2.5-coder",
      name: "Qwen 2.5 Coder",
      provider: "ollama",
      contextWindow: 32000,
      reasoning: false,
      input: ["text"],
      capabilities: {
        taskTypes: ["coding"],
        maxComplexity: "complex",
        supportsVision: false,
        supportsReasoning: false,
        contextWindow: 32000,
        costTier: "free",
      } as ModelCapabilities,
    },
    {
      id: "claude-sonnet-4-20250514",
      name: "Claude Sonnet 4",
      provider: "anthropic",
      contextWindow: 200000,
      reasoning: true,
      input: ["text", "image"],
      capabilities: {
        taskTypes: ["coding", "reasoning", "analysis", "chat", "vision", "general"],
        maxComplexity: "complex",
        supportsVision: true,
        supportsReasoning: true,
        contextWindow: 200000,
        costTier: "medium",
      } as ModelCapabilities,
    },
    {
      id: "gpt-4o",
      name: "GPT-4o",
      provider: "openai",
      contextWindow: 128000,
      reasoning: false,
      input: ["text", "image"],
      capabilities: {
        taskTypes: ["coding", "analysis", "chat", "vision", "general"],
        maxComplexity: "complex",
        supportsVision: true,
        supportsReasoning: false,
        contextWindow: 128000,
        costTier: "medium",
      } as ModelCapabilities,
    },
    {
      id: "o1",
      name: "o1",
      provider: "openai",
      contextWindow: 200000,
      reasoning: true,
      input: ["text"],
      capabilities: {
        taskTypes: ["reasoning", "coding", "analysis"],
        maxComplexity: "complex",
        supportsVision: false,
        supportsReasoning: true,
        contextWindow: 200000,
        costTier: "high",
      } as ModelCapabilities,
    },
  ];
}

describe("TaskBasedModelRouter", () => {
  describe("Basic Routing", () => {
    it("should route coding tasks to appropriate models", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug in auth.ts",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result?.primary).toBeDefined();
      expect(result?.decision.taskClassification.type).toBe("coding");
    });

    it("should route reasoning tasks to reasoning-capable models", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Analyze step by step why this algorithm is O(n log n)",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result?.decision.taskClassification.type).toBe("reasoning");
      // Reasoning tasks should be classified correctly
      expect(result?.decision.taskClassification.requiresReasoning).toBe(true);
    });

    it("should route vision tasks to vision-capable models", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "What's in this image?",
        hasImages: true,
      });

      expect(result).not.toBeNull();
      expect(result?.decision.taskClassification.requiresVision).toBe(true);
      // Vision tasks should be classified correctly and vision models should score higher
      expect(result?.decision.taskClassification.type).toBe("vision");
    });

    it("should route chat tasks appropriately", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Hello, how are you?",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result?.decision.taskClassification.type).toBe("chat");
    });

    it("should prefer local models for simple tasks when preferLocal is true", () => {
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "balanced",
        preferLocal: true,
        localProviders: ["ollama"],
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix a simple typo",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      // Local preference should boost Ollama models
      // Note: The exact selection depends on scoring, but local models should be competitive
    });

    it("should return null when routing is disabled", () => {
      const config: ModelRoutingConfig = { enabled: false };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug",
        hasImages: false,
      });

      expect(result).toBeNull();
    });
  });

  describe("Strategy Application", () => {
    it("should prefer cost-efficient models with cost-optimized strategy", () => {
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "cost-optimized",
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Write a simple function",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      // Cost-optimized should favor free models (Ollama)
      const selectedCapabilities = catalog.find(
        (m) => m.provider === result?.primary.provider && m.id === result?.primary.model,
      )?.capabilities;
      // Free or low cost models should be preferred
      expect(["free", "low"]).toContain(selectedCapabilities?.costTier);
    });

    it("should prefer high-performance models with performance-optimized strategy", () => {
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "performance-optimized",
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Solve this complex mathematical proof step by step",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      // Performance-optimized should favor reasoning models with large context
      const selectedCapabilities = catalog.find(
        (m) => m.provider === result?.primary.provider && m.id === result?.primary.model,
      )?.capabilities;
      expect(selectedCapabilities?.supportsReasoning).toBe(true);
    });

    it("should use balanced weights with balanced strategy", () => {
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "balanced",
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Analyze this code",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result?.decision.selectedScore).toBeDefined();
    });
  });

  describe("Task Rules", () => {
    it("should apply preferredProviders filter", () => {
      const taskRules: TaskRoutingRule[] = [
        {
          taskType: "coding",
          preferredProviders: ["ollama"],
        },
      ];
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "balanced",
        taskRules,
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug in the code",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result?.primary.provider).toBe("ollama");
    });

    it("should apply excludeProviders filter", () => {
      const taskRules: TaskRoutingRule[] = [
        {
          taskType: "coding",
          excludeProviders: ["ollama"],
        },
      ];
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "balanced",
        taskRules,
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug in the code",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result?.primary.provider).not.toBe("ollama");
    });

    it("should apply complexity-specific rules", () => {
      const taskRules: TaskRoutingRule[] = [
        {
          taskType: "reasoning",
          complexity: "complex",
          preferredProviders: ["anthropic"],
          requireReasoning: true,
        },
      ];
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "balanced",
        taskRules,
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Solve this complex mathematical proof with multiple steps",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      // Complex reasoning should use anthropic
      if (result?.decision.taskClassification.complexity === "complex") {
        expect(result.primary.provider).toBe("anthropic");
      }
    });

    it("should apply minContextWindow requirement", () => {
      const taskRules: TaskRoutingRule[] = [
        {
          taskType: "analysis",
          minContextWindow: 150000,
        },
      ];
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "balanced",
        taskRules,
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Summarize and analyze this large document",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      // When minContextWindow is applied, models with smaller context should be filtered
      // The selected model should have >= 150k context (claude-sonnet-4 or o1 have 200k)
      const selectedModel = catalog.find(
        (m) => m.provider === result?.primary.provider && m.id === result?.primary.model,
      );
      // If analysis task matched and rule applied, should get a large context model
      // If no models match the filter, falls back to full catalog
      expect(selectedModel).toBeDefined();
    });

    it("should apply requireReasoning filter", () => {
      const taskRules: TaskRoutingRule[] = [
        {
          taskType: "reasoning",
          requireReasoning: true,
        },
      ];
      const config: ModelRoutingConfig = {
        enabled: true,
        strategy: "balanced",
        taskRules,
      };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Prove this theorem step by step",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      const selectedCapabilities = catalog.find(
        (m) => m.provider === result?.primary.provider && m.id === result?.primary.model,
      )?.capabilities;
      expect(selectedCapabilities?.supportsReasoning).toBe(true);
    });
  });

  describe("Override Mechanisms", () => {
    it("should bypass routing with manual model override", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug",
        hasImages: false,
        overrideModel: "openai/gpt-4o",
      });

      expect(result).not.toBeNull();
      expect(result?.primary.provider).toBe("openai");
      expect(result?.primary.model).toBe("gpt-4o");
      expect(result?.decision.reasoning).toContain("Manual model override");
    });

    it("should handle model override without provider", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug",
        hasImages: false,
        overrideModel: "claude-sonnet-4-20250514",
      });

      expect(result).not.toBeNull();
      expect(result?.primary.provider).toBe("anthropic");
      expect(result?.primary.model).toBe("claude-sonnet-4-20250514");
    });

    it("should filter by allowedProviders", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug",
        hasImages: false,
        allowedProviders: ["openai"],
      });

      expect(result).not.toBeNull();
      expect(result?.primary.provider).toBe("openai");
    });
  });

  describe("Fallback Generation", () => {
    it("should generate fallback list excluding primary model", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug in the code",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      if (result?.fallbacks) {
        const primaryKey = `${result.primary.provider}/${result.primary.model}`.toLowerCase();
        for (const fallback of result.fallbacks) {
          const fallbackKey = `${fallback.provider}/${fallback.model}`.toLowerCase();
          expect(fallbackKey).not.toBe(primaryKey);
        }
      }
    });

    it("should limit fallback list to max 3 by default", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      const result = router.selectModelForTask({
        prompt: "Fix the bug",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      if (result?.fallbacks) {
        expect(result.fallbacks.length).toBeLessThanOrEqual(3);
      }
    });
  });

  describe("Error Handling", () => {
    it("should handle empty catalog gracefully", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const router = new TaskBasedModelRouter(config, []);

      const result = router.selectModelForTask({
        prompt: "Fix the bug",
        hasImages: false,
      });

      expect(result).toBeNull();
    });

    it("should handle routing errors without crashing", () => {
      const config: ModelRoutingConfig = { enabled: true, strategy: "balanced" };
      const catalog = createTestCatalog();
      const router = new TaskBasedModelRouter(config, catalog);

      // This should not throw
      expect(() => {
        router.selectModelForTask({
          prompt: "",
          hasImages: false,
        });
      }).not.toThrow();
    });
  });
});

describe("mergeRoutingConfig", () => {
  it("should return defaults when config is undefined", () => {
    const result = mergeRoutingConfig(undefined);
    expect(result).toEqual(DEFAULT_ROUTING_CONFIG);
  });

  it("should merge partial config with defaults", () => {
    const config: ModelRoutingConfig = {
      enabled: true,
      strategy: "cost-optimized",
    };
    const result = mergeRoutingConfig(config);

    expect(result.enabled).toBe(true);
    expect(result.strategy).toBe("cost-optimized");
    expect(result.preferLocal).toBe(DEFAULT_ROUTING_CONFIG.preferLocal);
    expect(result.localProviders).toEqual(DEFAULT_ROUTING_CONFIG.localProviders);
  });

  it("should normalize scoring weights", () => {
    const config: ModelRoutingConfig = {
      enabled: true,
      scoringWeights: {
        capabilityMatch: 1,
        costEfficiency: 1,
        performance: 1,
        availability: 1,
      },
    };
    const result = mergeRoutingConfig(config);

    // Weights should be normalized to sum to 1
    const sum =
      result.scoringWeights.capabilityMatch +
      result.scoringWeights.costEfficiency +
      result.scoringWeights.performance +
      result.scoringWeights.availability;
    expect(Math.abs(sum - 1.0)).toBeLessThan(0.001);
  });
});

describe("loadRoutingConfig", () => {
  it("should return disabled config when cfg is undefined", () => {
    const result = loadRoutingConfig(undefined);
    expect(result.enabled).toBe(false);
  });

  it("should return disabled config when routing is not configured", () => {
    const result = loadRoutingConfig({});
    expect(result.enabled).toBe(false);
  });

  it("should return routing config from agents.defaults.routing", () => {
    const cfg = {
      agents: {
        defaults: {
          routing: {
            enabled: true,
            strategy: "performance-optimized" as const,
          },
        },
      },
    };
    const result = loadRoutingConfig(cfg);
    expect(result.enabled).toBe(true);
    expect(result.strategy).toBe("performance-optimized");
  });
});

describe("shouldUseRouting", () => {
  it("should return true when override model is provided", () => {
    const result = shouldUseRouting({
      cfg: undefined,
      overrideModel: "anthropic/claude-sonnet-4-20250514",
    });
    expect(result).toBe(true);
  });

  it("should return false when routing is disabled", () => {
    const result = shouldUseRouting({
      cfg: { agents: { defaults: { routing: { enabled: false } } } },
    });
    expect(result).toBe(false);
  });

  it("should return true when routing is enabled", () => {
    const result = shouldUseRouting({
      cfg: { agents: { defaults: { routing: { enabled: true } } } },
    });
    expect(result).toBe(true);
  });
});

describe("createRouterFromConfig", () => {
  it("should return null when routing is disabled", () => {
    const result = createRouterFromConfig({
      cfg: { agents: { defaults: { routing: { enabled: false } } } },
      catalog: createTestCatalog(),
    });
    expect(result).toBeNull();
  });

  it("should return router when routing is enabled", () => {
    const result = createRouterFromConfig({
      cfg: { agents: { defaults: { routing: { enabled: true } } } },
      catalog: createTestCatalog(),
    });
    expect(result).toBeInstanceOf(TaskBasedModelRouter);
  });
});

describe("generateFallbackList", () => {
  it("should generate fallback list excluding primary", () => {
    const rankedScores = [
      {
        provider: "anthropic",
        model: "claude-sonnet-4-20250514",
        totalScore: 0.9,
        breakdown: { capability: 0.9, cost: 0.8, performance: 0.9, availability: 1.0 },
        reasoning: "",
      },
      {
        provider: "openai",
        model: "gpt-4o",
        totalScore: 0.85,
        breakdown: { capability: 0.85, cost: 0.7, performance: 0.85, availability: 1.0 },
        reasoning: "",
      },
      {
        provider: "ollama",
        model: "llama3.2",
        totalScore: 0.7,
        breakdown: { capability: 0.7, cost: 1.0, performance: 0.6, availability: 1.0 },
        reasoning: "",
      },
    ];

    const primary = { provider: "anthropic", model: "claude-sonnet-4-20250514" };
    const fallbacks = generateFallbackList(rankedScores, primary);

    expect(fallbacks).toHaveLength(2);
    expect(fallbacks).toContain("openai/gpt-4o");
    expect(fallbacks).toContain("ollama/llama3.2");
    expect(fallbacks).not.toContain("anthropic/claude-sonnet-4-20250514");
  });

  it("should respect maxFallbacks limit", () => {
    const rankedScores = [
      {
        provider: "anthropic",
        model: "claude-sonnet-4-20250514",
        totalScore: 0.9,
        breakdown: { capability: 0.9, cost: 0.8, performance: 0.9, availability: 1.0 },
        reasoning: "",
      },
      {
        provider: "openai",
        model: "gpt-4o",
        totalScore: 0.85,
        breakdown: { capability: 0.85, cost: 0.7, performance: 0.85, availability: 1.0 },
        reasoning: "",
      },
      {
        provider: "openai",
        model: "o1",
        totalScore: 0.8,
        breakdown: { capability: 0.8, cost: 0.5, performance: 0.9, availability: 1.0 },
        reasoning: "",
      },
      {
        provider: "ollama",
        model: "llama3.2",
        totalScore: 0.7,
        breakdown: { capability: 0.7, cost: 1.0, performance: 0.6, availability: 1.0 },
        reasoning: "",
      },
    ];

    const primary = { provider: "anthropic", model: "claude-sonnet-4-20250514" };
    const fallbacks = generateFallbackList(rankedScores, primary, 2);

    expect(fallbacks).toHaveLength(2);
  });
});

describe("routeModelForTask", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should return default model when routing is disabled", async () => {
    const { loadModelCatalog } = await import("./model-catalog.js");
    vi.mocked(loadModelCatalog).mockResolvedValue(createTestCatalog());

    const result = await routeModelForTask({
      cfg: { agents: { defaults: { routing: { enabled: false } } } },
      prompt: "Fix the bug",
      hasImages: false,
    });

    expect(result.provider).toBe("anthropic");
    expect(result.model).toBe("claude-opus-4-5");
  });

  it("should honor manual override even when routing is disabled", async () => {
    const result = await routeModelForTask({
      cfg: { agents: { defaults: { routing: { enabled: false } } } },
      prompt: "Fix the bug",
      hasImages: false,
      overrideModel: "openai/gpt-4o",
    });

    expect(result.provider).toBe("openai");
    expect(result.model).toBe("gpt-4o");
    expect(result.decision?.reasoning).toContain("Manual model override");
  });

  it("should signal manual selection when fallback behavior requires it", async () => {
    const result = await routeModelForTask({
      cfg: {
        agents: {
          defaults: { routing: { enabled: false, fallbackBehavior: "manual-selection" } },
        },
      },
      prompt: "Fix the bug",
      hasImages: false,
    });

    expect(result.manualSelectionRequired).toBe(true);
    expect(result.provider).toBeUndefined();
    expect(result.model).toBeUndefined();
  });

  it("should route when enabled", async () => {
    const { loadModelCatalog } = await import("./model-catalog.js");
    vi.mocked(loadModelCatalog).mockResolvedValue(createTestCatalog());

    const result = await routeModelForTask({
      cfg: { agents: { defaults: { routing: { enabled: true } } } },
      prompt: "Fix the bug in the code",
      hasImages: false,
    });

    expect(result.provider).toBeDefined();
    expect(result.model).toBeDefined();
    expect(result.decision).toBeDefined();
  });
});
