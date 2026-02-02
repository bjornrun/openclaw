import { describe, expect, it } from "vitest";
import type { ModelScoringWeights } from "../config/types.models.js";
import type { ModelCatalogEntry } from "./model-catalog.js";
import type { TaskClassification } from "./task-classifier.js";
import {
  DEFAULT_SCORING_WEIGHTS,
  filterModelsByCapabilities,
  normalizeWeights,
  rankModels,
  scoreAvailability,
  scoreCapabilityMatch,
  scoreCostEfficiency,
  scoreModel,
  scorePerformance,
  selectBestModelForTask,
} from "./model-scorer.js";

// Test fixtures: mock model catalog entries

const ollamaCodingModel: ModelCatalogEntry = {
  id: "codellama:7b",
  name: "Code Llama 7B",
  provider: "ollama",
  contextWindow: 128_000,
  reasoning: false,
  input: ["text"],
  capabilities: {
    taskTypes: ["coding", "general", "chat"],
    maxComplexity: "moderate",
    supportsVision: false,
    supportsReasoning: false,
    contextWindow: 128_000,
    costTier: "free",
  },
};

const ollamaGeneralModel: ModelCatalogEntry = {
  id: "llama3:8b",
  name: "Llama 3 8B",
  provider: "ollama",
  contextWindow: 8_000,
  reasoning: false,
  input: ["text"],
  capabilities: {
    taskTypes: ["general", "chat"],
    maxComplexity: "simple",
    supportsVision: false,
    supportsReasoning: false,
    contextWindow: 8_000,
    costTier: "free",
  },
};

const cloudReasoningModel: ModelCatalogEntry = {
  id: "claude-3-opus",
  name: "Claude 3 Opus",
  provider: "anthropic",
  contextWindow: 200_000,
  reasoning: true,
  input: ["text", "image"],
  capabilities: {
    taskTypes: ["coding", "reasoning", "analysis", "general", "chat", "vision"],
    maxComplexity: "complex",
    supportsVision: true,
    supportsReasoning: true,
    contextWindow: 200_000,
    costTier: "high",
  },
};

const cloudVisionModel: ModelCatalogEntry = {
  id: "gpt-4-vision",
  name: "GPT-4 Vision",
  provider: "openai",
  contextWindow: 128_000,
  reasoning: false,
  input: ["text", "image"],
  capabilities: {
    taskTypes: ["vision", "general", "chat", "analysis"],
    maxComplexity: "moderate",
    supportsVision: true,
    supportsReasoning: false,
    contextWindow: 128_000,
    costTier: "medium",
  },
};

const cloudGeneralModel: ModelCatalogEntry = {
  id: "gpt-3.5-turbo",
  name: "GPT-3.5 Turbo",
  provider: "openai",
  contextWindow: 32_000,
  reasoning: false,
  input: ["text"],
  capabilities: {
    taskTypes: ["general", "chat", "coding"],
    maxComplexity: "moderate",
    supportsVision: false,
    supportsReasoning: false,
    contextWindow: 32_000,
    costTier: "low",
  },
};

const allModels: ModelCatalogEntry[] = [
  ollamaCodingModel,
  ollamaGeneralModel,
  cloudReasoningModel,
  cloudVisionModel,
  cloudGeneralModel,
];

// Task classification fixtures

const simpleCodingTask: TaskClassification = {
  type: "coding",
  complexity: "simple",
  requiresVision: false,
  requiresReasoning: false,
  confidence: 0.9,
};

const complexReasoningTask: TaskClassification = {
  type: "reasoning",
  complexity: "complex",
  requiresVision: false,
  requiresReasoning: true,
  confidence: 0.85,
};

const visionTask: TaskClassification = {
  type: "vision",
  complexity: "moderate",
  requiresVision: true,
  requiresReasoning: false,
  confidence: 0.95,
};

const largeContextTask: TaskClassification = {
  type: "analysis",
  complexity: "complex",
  requiresVision: false,
  requiresReasoning: true,
  estimatedContextTokens: 150_000,
  confidence: 0.8,
};

const simpleChatTask: TaskClassification = {
  type: "chat",
  complexity: "simple",
  requiresVision: false,
  requiresReasoning: false,
  confidence: 0.7,
};

describe("model-scorer", () => {
  describe("normalizeWeights", () => {
    it("returns default weights when sum is zero", () => {
      const weights: ModelScoringWeights = {
        capabilityMatch: 0,
        costEfficiency: 0,
        performance: 0,
        availability: 0,
      };
      expect(normalizeWeights(weights)).toEqual(DEFAULT_SCORING_WEIGHTS);
    });

    it("returns weights unchanged when sum is 1.0", () => {
      const weights: ModelScoringWeights = {
        capabilityMatch: 0.5,
        costEfficiency: 0.2,
        performance: 0.2,
        availability: 0.1,
      };
      expect(normalizeWeights(weights)).toEqual(weights);
    });

    it("normalizes weights that do not sum to 1.0", () => {
      const weights: ModelScoringWeights = {
        capabilityMatch: 2,
        costEfficiency: 1,
        performance: 0.5,
        availability: 0.5,
      };
      const normalized = normalizeWeights(weights);
      const sum =
        normalized.capabilityMatch +
        normalized.costEfficiency +
        normalized.performance +
        normalized.availability;
      expect(sum).toBeCloseTo(1.0, 5);
      expect(normalized.capabilityMatch).toBeCloseTo(0.5, 5);
    });
  });

  describe("scoreCapabilityMatch", () => {
    it("scores high for matching task type", () => {
      const score = scoreCapabilityMatch(ollamaCodingModel, simpleCodingTask);
      expect(score).toBeGreaterThan(0.8);
    });

    it("scores lower for non-matching task type", () => {
      const score = scoreCapabilityMatch(ollamaGeneralModel, simpleCodingTask);
      expect(score).toBeLessThan(scoreCapabilityMatch(ollamaCodingModel, simpleCodingTask));
    });

    it("penalizes models that cannot handle task complexity", () => {
      const simpleModelScore = scoreCapabilityMatch(ollamaGeneralModel, complexReasoningTask);
      const complexModelScore = scoreCapabilityMatch(cloudReasoningModel, complexReasoningTask);
      expect(complexModelScore).toBeGreaterThan(simpleModelScore);
    });

    it("requires vision support for vision tasks", () => {
      const visionModelScore = scoreCapabilityMatch(cloudVisionModel, visionTask);
      const nonVisionModelScore = scoreCapabilityMatch(ollamaCodingModel, visionTask);
      expect(visionModelScore).toBeGreaterThan(nonVisionModelScore);
    });

    it("prefers reasoning models for reasoning tasks", () => {
      const reasoningModelScore = scoreCapabilityMatch(cloudReasoningModel, complexReasoningTask);
      const nonReasoningModelScore = scoreCapabilityMatch(cloudGeneralModel, complexReasoningTask);
      expect(reasoningModelScore).toBeGreaterThan(nonReasoningModelScore);
    });

    it("penalizes models with insufficient context window", () => {
      const largeContextModelScore = scoreCapabilityMatch(cloudReasoningModel, largeContextTask);
      const smallContextModelScore = scoreCapabilityMatch(ollamaGeneralModel, largeContextTask);
      expect(largeContextModelScore).toBeGreaterThan(smallContextModelScore);
    });
  });

  describe("scoreCostEfficiency", () => {
    it("scores free tier (Ollama) as 1.0", () => {
      expect(scoreCostEfficiency(ollamaCodingModel)).toBe(1.0);
      expect(scoreCostEfficiency(ollamaGeneralModel)).toBe(1.0);
    });

    it("scores low tier as 0.7", () => {
      expect(scoreCostEfficiency(cloudGeneralModel)).toBe(0.7);
    });

    it("scores medium tier as 0.4", () => {
      expect(scoreCostEfficiency(cloudVisionModel)).toBe(0.4);
    });

    it("scores high tier as 0.2", () => {
      expect(scoreCostEfficiency(cloudReasoningModel)).toBe(0.2);
    });
  });

  describe("scorePerformance", () => {
    it("scores higher for larger context windows", () => {
      const largeContextScore = scorePerformance(cloudReasoningModel, simpleChatTask);
      const smallContextScore = scorePerformance(ollamaGeneralModel, simpleChatTask);
      expect(largeContextScore).toBeGreaterThan(smallContextScore);
    });

    it("gives reasoning bonus for complex tasks", () => {
      const reasoningModelScore = scorePerformance(cloudReasoningModel, complexReasoningTask);
      const nonReasoningModelScore = scorePerformance(cloudVisionModel, complexReasoningTask);
      expect(reasoningModelScore).toBeGreaterThan(nonReasoningModelScore);
    });

    it("gives reasoning bonus for reasoning tasks", () => {
      const reasoningModelScore = scorePerformance(cloudReasoningModel, complexReasoningTask);
      expect(reasoningModelScore).toBeGreaterThan(0.8);
    });
  });

  describe("scoreAvailability", () => {
    it("returns 1.0 by default", () => {
      expect(scoreAvailability(ollamaCodingModel)).toBe(1.0);
    });

    it("uses provided health score", () => {
      expect(scoreAvailability(ollamaCodingModel, 0.5)).toBe(0.5);
    });

    it("clamps health score to 0-1 range", () => {
      expect(scoreAvailability(ollamaCodingModel, 1.5)).toBe(1.0);
      expect(scoreAvailability(ollamaCodingModel, -0.5)).toBe(0.0);
    });
  });

  describe("scoreModel", () => {
    it("returns complete score with breakdown", () => {
      const score = scoreModel({
        model: ollamaCodingModel,
        taskClassification: simpleCodingTask,
      });

      expect(score.provider).toBe("ollama");
      expect(score.model).toBe("codellama:7b");
      expect(score.totalScore).toBeGreaterThan(0);
      expect(score.totalScore).toBeLessThanOrEqual(1);
      expect(score.breakdown.capability).toBeGreaterThan(0);
      expect(score.breakdown.cost).toBe(1.0);
      expect(score.breakdown.performance).toBeGreaterThan(0);
      expect(score.breakdown.availability).toBe(1.0);
      expect(score.reasoning).toContain("coding");
    });

    it("uses default weights when not provided", () => {
      const score = scoreModel({
        model: ollamaCodingModel,
        taskClassification: simpleCodingTask,
      });

      const expectedTotal =
        score.breakdown.capability * DEFAULT_SCORING_WEIGHTS.capabilityMatch +
        score.breakdown.cost * DEFAULT_SCORING_WEIGHTS.costEfficiency +
        score.breakdown.performance * DEFAULT_SCORING_WEIGHTS.performance +
        score.breakdown.availability * DEFAULT_SCORING_WEIGHTS.availability;

      expect(score.totalScore).toBeCloseTo(expectedTotal, 5);
    });

    it("applies custom weights", () => {
      const customWeights: ModelScoringWeights = {
        capabilityMatch: 0.1,
        costEfficiency: 0.7,
        performance: 0.1,
        availability: 0.1,
      };

      const defaultScore = scoreModel({
        model: cloudReasoningModel,
        taskClassification: simpleCodingTask,
      });

      const customScore = scoreModel({
        model: cloudReasoningModel,
        taskClassification: simpleCodingTask,
        weights: customWeights,
      });

      // With high cost weight, expensive model should score lower
      expect(customScore.totalScore).toBeLessThan(defaultScore.totalScore);
    });
  });

  describe("rankModels", () => {
    it("returns empty array for empty catalog with warning", () => {
      const ranked = rankModels([], simpleCodingTask);
      expect(ranked).toEqual([]);
    });

    it("sorts models by total score descending", () => {
      const ranked = rankModels(allModels, simpleCodingTask);
      for (let i = 1; i < ranked.length; i++) {
        expect(ranked[i - 1].totalScore).toBeGreaterThanOrEqual(ranked[i].totalScore);
      }
    });

    it("prefers free Ollama models for simple coding tasks", () => {
      const ranked = rankModels(allModels, simpleCodingTask);
      // Ollama coding model should rank high due to free cost + capability match
      const ollamaRank = ranked.findIndex(
        (s) => s.provider === "ollama" && s.model === "codellama:7b",
      );
      expect(ollamaRank).toBeLessThan(3);
    });

    it("prefers cloud reasoning models for complex reasoning tasks", () => {
      const ranked = rankModels(allModels, complexReasoningTask);
      // Cloud reasoning model should rank first despite cost
      expect(ranked[0].provider).toBe("anthropic");
      expect(ranked[0].model).toBe("claude-3-opus");
    });

    it("only selects vision models for vision tasks", () => {
      const ranked = rankModels(allModels, visionTask);
      // Vision models should rank higher
      const topTwo = ranked.slice(0, 2);
      const hasVisionModel = topTwo.some(
        (s) => s.model === "claude-3-opus" || s.model === "gpt-4-vision",
      );
      expect(hasVisionModel).toBe(true);
    });

    it("prefers models with larger context for large context tasks", () => {
      const ranked = rankModels(allModels, largeContextTask);
      // Model with 200k context should rank high
      expect(ranked[0].model).toBe("claude-3-opus");
    });

    it("uses tie-breakers for equal scores", () => {
      // Create two models with similar capabilities but different costs
      const model1: ModelCatalogEntry = {
        ...ollamaCodingModel,
        id: "model-a",
        provider: "test",
      };
      const model2: ModelCatalogEntry = {
        ...ollamaCodingModel,
        id: "model-b",
        provider: "test",
        capabilities: {
          ...ollamaCodingModel.capabilities!,
          costTier: "low",
        },
      };
      const ranked = rankModels([model1, model2], simpleCodingTask);
      // Free model should come first as tie-breaker
      expect(ranked[0].model).toBe("model-a");
    });
  });

  describe("filterModelsByCapabilities", () => {
    it("filters by vision requirement", () => {
      const filtered = filterModelsByCapabilities(allModels, { requiresVision: true });
      expect(filtered.every((m) => m.capabilities?.supportsVision)).toBe(true);
      expect(filtered.length).toBe(2);
    });

    it("filters by reasoning requirement", () => {
      const filtered = filterModelsByCapabilities(allModels, { requiresReasoning: true });
      expect(filtered.every((m) => m.capabilities?.supportsReasoning)).toBe(true);
      expect(filtered.length).toBe(1);
    });

    it("filters by minimum context window", () => {
      const filtered = filterModelsByCapabilities(allModels, { minContextWindow: 100_000 });
      expect(filtered.every((m) => (m.capabilities?.contextWindow ?? 0) >= 100_000)).toBe(true);
      expect(filtered.length).toBe(3);
    });

    it("filters by allowed providers", () => {
      const filtered = filterModelsByCapabilities(allModels, {
        allowedProviders: ["ollama", "openai"],
      });
      expect(filtered.every((m) => ["ollama", "openai"].includes(m.provider))).toBe(true);
      expect(filtered.length).toBe(4);
    });

    it("combines multiple filters", () => {
      const filtered = filterModelsByCapabilities(allModels, {
        requiresVision: true,
        minContextWindow: 100_000,
      });
      expect(filtered.length).toBe(2);
    });

    it("returns all models when no filters specified", () => {
      const filtered = filterModelsByCapabilities(allModels, {});
      expect(filtered.length).toBe(allModels.length);
    });
  });

  describe("explainScore", () => {
    it("generates human-readable reasoning string", () => {
      const score = scoreModel({
        model: ollamaCodingModel,
        taskClassification: simpleCodingTask,
      });

      expect(score.reasoning).toContain("ollama/codellama:7b");
      expect(score.reasoning).toContain("coding");
      expect(score.reasoning).toContain("capability match");
      expect(score.reasoning).toContain("cost efficiency");
      expect(score.reasoning).toContain("performance");
      expect(score.reasoning).toContain("availability");
    });

    it("includes percentage values", () => {
      const score = scoreModel({
        model: cloudReasoningModel,
        taskClassification: complexReasoningTask,
      });

      // Should contain percentage values
      expect(score.reasoning).toMatch(/\d+%/);
    });
  });

  describe("selectBestModelForTask", () => {
    it("returns null for empty catalog", () => {
      const result = selectBestModelForTask({
        catalog: [],
        prompt: "Fix the bug",
        hasImages: false,
      });
      expect(result).toBeNull();
    });

    it("classifies task and returns best model", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Debug this TypeScript function",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result!.task.type).toBe("coding");
      expect(result!.score).toBeDefined();
      expect(result!.allRanked.length).toBe(allModels.length);
    });

    it("selects vision model when images are present", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "What is in this image?",
        hasImages: true,
      });

      expect(result).not.toBeNull();
      expect(result!.task.requiresVision).toBe(true);
      // Top model should support vision
      const topModel = allModels.find(
        (m) => m.provider === result!.score.provider && m.id === result!.score.model,
      );
      expect(topModel?.capabilities?.supportsVision).toBe(true);
    });

    it("applies task classification hints", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Hello",
        hasImages: false,
        hints: { taskType: "reasoning", complexity: "complex" },
      });

      expect(result).not.toBeNull();
      expect(result!.task.type).toBe("reasoning");
      expect(result!.task.complexity).toBe("complex");
    });

    it("applies custom weights", () => {
      const costFocusedWeights: ModelScoringWeights = {
        capabilityMatch: 0.1,
        costEfficiency: 0.7,
        performance: 0.1,
        availability: 0.1,
      };

      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Fix this simple typo in the function",
        hasImages: false,
        weights: costFocusedWeights,
      });

      expect(result).not.toBeNull();
      // With cost-focused weights, free Ollama models should rank higher
      expect(result!.score.provider).toBe("ollama");
    });

    it("falls back to all models when no models match hard requirements", () => {
      // Create a catalog with no vision models
      const noVisionCatalog = [ollamaCodingModel, ollamaGeneralModel, cloudGeneralModel];

      const result = selectBestModelForTask({
        catalog: noVisionCatalog,
        prompt: "Analyze this screenshot",
        hasImages: true,
      });

      // Should still return a result (best effort)
      expect(result).not.toBeNull();
      expect(result!.allRanked.length).toBe(noVisionCatalog.length);
    });

    it("returns all ranked models for fallback purposes", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Write a function",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      expect(result!.allRanked.length).toBeGreaterThan(1);
      // All ranked should be sorted by score
      for (let i = 1; i < result!.allRanked.length; i++) {
        expect(result!.allRanked[i - 1].totalScore).toBeGreaterThanOrEqual(
          result!.allRanked[i].totalScore,
        );
      }
    });
  });

  describe("integration scenarios", () => {
    it("simple coding task prefers free Ollama coding model over expensive cloud model", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Fix this simple typo in the function",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      // For simple coding, Ollama coding model should be preferred
      const topScore = result!.score;
      expect(topScore.breakdown.cost).toBe(1.0); // Free model
    });

    it("complex reasoning task prefers cloud reasoning model despite cost", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Analyze this complex mathematical proof step by step and explain the reasoning",
        hasImages: false,
      });

      expect(result).not.toBeNull();
      // For complex reasoning, cloud reasoning model should be preferred
      expect(result!.score.provider).toBe("anthropic");
    });

    it("vision task only selects models with vision support", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Look at this image and describe what you see",
        hasImages: true,
      });

      expect(result).not.toBeNull();
      const topModel = allModels.find(
        (m) => m.provider === result!.score.provider && m.id === result!.score.model,
      );
      expect(topModel?.capabilities?.supportsVision).toBe(true);
    });

    it("large context task prefers models with larger context windows", () => {
      const result = selectBestModelForTask({
        catalog: allModels,
        prompt: "Analyze this entire codebase comprehensively",
        hasImages: false,
        hints: { complexity: "complex" },
      });

      expect(result).not.toBeNull();
      // Model with largest context window should rank high
      const topModel = allModels.find(
        (m) => m.provider === result!.score.provider && m.id === result!.score.model,
      );
      expect(topModel?.contextWindow).toBeGreaterThanOrEqual(128_000);
    });
  });

  describe("edge cases", () => {
    it("handles model without pre-computed capabilities", () => {
      const modelWithoutCaps: ModelCatalogEntry = {
        id: "test-model",
        name: "Test Model",
        provider: "test",
        contextWindow: 32_000,
        reasoning: true,
        input: ["text", "image"],
      };

      const score = scoreModel({
        model: modelWithoutCaps,
        taskClassification: simpleCodingTask,
      });

      expect(score.totalScore).toBeGreaterThan(0);
    });

    it("handles task with undefined estimatedContextTokens", () => {
      const taskWithoutContext: TaskClassification = {
        type: "general",
        complexity: "simple",
        requiresVision: false,
        requiresReasoning: false,
        confidence: 0.5,
      };

      const score = scoreCapabilityMatch(ollamaCodingModel, taskWithoutContext);
      expect(score).toBeGreaterThan(0);
    });

    it("handles zero estimatedContextTokens", () => {
      const taskWithZeroContext: TaskClassification = {
        type: "general",
        complexity: "simple",
        requiresVision: false,
        requiresReasoning: false,
        estimatedContextTokens: 0,
        confidence: 0.5,
      };

      const score = scoreCapabilityMatch(ollamaCodingModel, taskWithZeroContext);
      expect(score).toBeGreaterThan(0);
    });
  });
});
