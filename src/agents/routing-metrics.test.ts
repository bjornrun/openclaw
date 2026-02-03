import { describe, expect, it, beforeEach } from "vitest";
import type { RoutingDecision } from "./task-router.js";
import {
  createRoutingMetrics,
  createNoopRoutingMetrics,
  getGlobalRoutingMetrics,
  resetGlobalRoutingMetrics,
  type RoutingMetrics,
} from "./routing-metrics.js";

function createMockDecision(overrides: Partial<RoutingDecision> = {}): RoutingDecision {
  return {
    strategy: "balanced",
    taskClassification: {
      type: "coding",
      complexity: "simple",
      requiresVision: false,
      requiresReasoning: false,
      confidence: 0.85,
    },
    selectedScore: {
      provider: "ollama",
      model: "llama3.2",
      totalScore: 0.75,
      breakdown: {
        capability: 0.8,
        cost: 1.0,
        performance: 0.6,
        availability: 1.0,
      },
      reasoning: "Selected for coding task",
    },
    allRankedScores: [
      {
        provider: "ollama",
        model: "llama3.2",
        totalScore: 0.75,
        breakdown: {
          capability: 0.8,
          cost: 1.0,
          performance: 0.6,
          availability: 1.0,
        },
        reasoning: "Selected for coding task",
      },
    ],
    reasoning: "Task-based routing: Selected ollama/llama3.2 for coding task",
    ...overrides,
  };
}

describe("createRoutingMetrics", () => {
  let metrics: RoutingMetrics;

  beforeEach(() => {
    metrics = createRoutingMetrics();
  });

  describe("recordDecision", () => {
    it("should increment total decisions", () => {
      const decision = createMockDecision();
      metrics.recordDecision(decision);

      const snapshot = metrics.getSnapshot();
      expect(snapshot.totalDecisions).toBe(1);
    });

    it("should track decisions by task type", () => {
      metrics.recordDecision(createMockDecision());
      metrics.recordDecision(
        createMockDecision({
          taskClassification: {
            type: "reasoning",
            complexity: "complex",
            requiresVision: false,
            requiresReasoning: true,
            confidence: 0.9,
          },
        }),
      );
      metrics.recordDecision(createMockDecision());

      const snapshot = metrics.getSnapshot();
      expect(snapshot.decisionsByTaskType.coding).toBe(2);
      expect(snapshot.decisionsByTaskType.reasoning).toBe(1);
    });

    it("should track decisions by strategy", () => {
      metrics.recordDecision(createMockDecision());
      metrics.recordDecision(createMockDecision({ strategy: "cost-optimized" }));

      const snapshot = metrics.getSnapshot();
      expect(snapshot.decisionsByStrategy.balanced).toBe(1);
      expect(snapshot.decisionsByStrategy["cost-optimized"]).toBe(1);
    });

    it("should track selections by provider", () => {
      metrics.recordDecision(createMockDecision());
      metrics.recordDecision(
        createMockDecision({
          selectedScore: {
            provider: "anthropic",
            model: "claude-sonnet-4",
            totalScore: 0.9,
            breakdown: {
              capability: 0.95,
              cost: 0.5,
              performance: 0.9,
              availability: 1.0,
            },
            reasoning: "Selected for complex task",
          },
        }),
      );

      const snapshot = metrics.getSnapshot();
      expect(snapshot.selectionsByProvider.ollama).toBe(1);
      expect(snapshot.selectionsByProvider.anthropic).toBe(1);
    });

    it("should track model selections with scores", () => {
      metrics.recordDecision(createMockDecision());
      metrics.recordDecision(createMockDecision());

      const snapshot = metrics.getSnapshot();
      const modelData = snapshot.modelSelections["ollama/llama3.2"];
      expect(modelData).toBeDefined();
      expect(modelData.count).toBe(2);
      expect(modelData.avgScore).toBeCloseTo(0.75);
    });
  });

  describe("recordTaskType", () => {
    it("should emit task classification events without incrementing decisions", () => {
      const events: Array<{ name: string; value: number }> = [];
      const metricsWithCallback = createRoutingMetrics((event) => {
        events.push({ name: event.name, value: event.value });
      });

      metricsWithCallback.recordTaskType("coding");
      metricsWithCallback.recordTaskType("reasoning");

      const snapshot = metricsWithCallback.getSnapshot();
      expect(snapshot.totalDecisions).toBe(0);
      expect(snapshot.decisionsByTaskType).toEqual({});
      expect(events).toContainEqual({ name: "routing.task_classified", value: 1 });
    });
  });

  describe("recordModelSelection", () => {
    it("should track model selections independently", () => {
      metrics.recordModelSelection("ollama", "llama3.2", 0.8);
      metrics.recordModelSelection("ollama", "llama3.2", 0.7);
      metrics.recordModelSelection("anthropic", "claude-sonnet-4", 0.9);

      const snapshot = metrics.getSnapshot();
      expect(snapshot.selectionsByProvider.ollama).toBe(2);
      expect(snapshot.selectionsByProvider.anthropic).toBe(1);
      expect(snapshot.modelSelections["ollama/llama3.2"].count).toBe(2);
      expect(snapshot.modelSelections["ollama/llama3.2"].avgScore).toBeCloseTo(0.75);
    });
  });

  describe("recordFallback", () => {
    it("should increment fallback counter", () => {
      metrics.recordFallback("anthropic", "claude-sonnet-4");
      metrics.recordFallback("openai", "gpt-4o");

      const snapshot = metrics.getSnapshot();
      expect(snapshot.fallbacksUsed).toBe(2);
    });
  });

  describe("recordOverride", () => {
    it("should increment override counter", () => {
      metrics.recordOverride("openai", "gpt-4o");

      const snapshot = metrics.getSnapshot();
      expect(snapshot.overridesUsed).toBe(1);
    });
  });

  describe("recordError", () => {
    it("should increment error counter", () => {
      metrics.recordError("Model not found");
      metrics.recordError("Routing failed");

      const snapshot = metrics.getSnapshot();
      expect(snapshot.errors).toBe(2);
    });
  });

  describe("getSnapshot", () => {
    it("should return current state", () => {
      metrics.recordDecision(createMockDecision());
      metrics.recordFallback("anthropic", "claude-sonnet-4");
      metrics.recordError("Test error");

      const snapshot = metrics.getSnapshot();
      expect(snapshot.totalDecisions).toBe(1);
      expect(snapshot.fallbacksUsed).toBe(1);
      expect(snapshot.errors).toBe(1);
      expect(snapshot.snapshotAt).toBeGreaterThan(0);
    });

    it("should include timestamp", () => {
      const before = Date.now();
      const snapshot = metrics.getSnapshot();
      const after = Date.now();

      expect(snapshot.snapshotAt).toBeGreaterThanOrEqual(before);
      expect(snapshot.snapshotAt).toBeLessThanOrEqual(after);
    });
  });

  describe("reset", () => {
    it("should reset all counters to zero", () => {
      metrics.recordDecision(createMockDecision());
      metrics.recordFallback("anthropic", "claude-sonnet-4");
      metrics.recordOverride("openai", "gpt-4o");
      metrics.recordError("Test error");

      metrics.reset();

      const snapshot = metrics.getSnapshot();
      expect(snapshot.totalDecisions).toBe(0);
      expect(snapshot.fallbacksUsed).toBe(0);
      expect(snapshot.overridesUsed).toBe(0);
      expect(snapshot.errors).toBe(0);
      expect(Object.keys(snapshot.decisionsByTaskType)).toHaveLength(0);
      expect(Object.keys(snapshot.selectionsByProvider)).toHaveLength(0);
      expect(Object.keys(snapshot.modelSelections)).toHaveLength(0);
    });
  });

  describe("onMetric callback", () => {
    it("should call callback for each metric event", () => {
      const events: Array<{ name: string; value: number }> = [];
      const metricsWithCallback = createRoutingMetrics((event) => {
        events.push({ name: event.name, value: event.value });
      });

      metricsWithCallback.recordDecision(createMockDecision());
      metricsWithCallback.recordFallback("anthropic", "claude-sonnet-4");
      metricsWithCallback.recordError("Test error");

      expect(events).toContainEqual({ name: "routing.decision", value: 1 });
      expect(events).toContainEqual({ name: "routing.fallback_used", value: 1 });
      expect(events).toContainEqual({ name: "routing.error", value: 1 });
    });
  });
});

describe("createNoopRoutingMetrics", () => {
  it("should return metrics that do nothing", () => {
    const metrics = createNoopRoutingMetrics();

    // These should not throw
    metrics.recordDecision(createMockDecision());
    metrics.recordTaskType("coding");
    metrics.recordModelSelection("ollama", "llama3.2", 0.8);
    metrics.recordFallback("anthropic", "claude-sonnet-4");
    metrics.recordOverride("openai", "gpt-4o");
    metrics.recordError("Test error");

    const snapshot = metrics.getSnapshot();
    expect(snapshot.totalDecisions).toBe(0);
    expect(snapshot.fallbacksUsed).toBe(0);
    expect(snapshot.errors).toBe(0);
  });

  it("should return empty snapshot with current timestamp", () => {
    const metrics = createNoopRoutingMetrics();
    const before = Date.now();
    const snapshot = metrics.getSnapshot();
    const after = Date.now();

    expect(snapshot.snapshotAt).toBeGreaterThanOrEqual(before);
    expect(snapshot.snapshotAt).toBeLessThanOrEqual(after);
  });
});

describe("getGlobalRoutingMetrics", () => {
  beforeEach(() => {
    resetGlobalRoutingMetrics();
  });

  it("should return the same instance on multiple calls", () => {
    const metrics1 = getGlobalRoutingMetrics();
    const metrics2 = getGlobalRoutingMetrics();

    expect(metrics1).toBe(metrics2);
  });

  it("should persist state across calls", () => {
    const metrics = getGlobalRoutingMetrics();
    metrics.recordDecision(createMockDecision());

    const metrics2 = getGlobalRoutingMetrics();
    const snapshot = metrics2.getSnapshot();

    expect(snapshot.totalDecisions).toBe(1);
  });
});

describe("resetGlobalRoutingMetrics", () => {
  it("should reset global metrics state", () => {
    const metrics = getGlobalRoutingMetrics();
    metrics.recordDecision(createMockDecision());

    resetGlobalRoutingMetrics();

    const snapshot = getGlobalRoutingMetrics().getSnapshot();
    expect(snapshot.totalDecisions).toBe(0);
  });
});
