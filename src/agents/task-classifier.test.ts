import { describe, expect, it } from "vitest";
import { classifyTask } from "./task-classifier.js";

describe("task-classifier", () => {
  it("detects coding tasks and infers simple complexity for debugging prompts", () => {
    const result = classifyTask("Debug this TypeScript function", false);

    expect(result.type).toBe("coding");
    expect(result.complexity).toBe("simple");
    expect(result.requiresReasoning).toBe(false);
    expect(result.requiresVision).toBe(false);
  });

  it("detects reasoning tasks and flags complex reasoning requirements", () => {
    const result = classifyTask(
      "Analyze this complex mathematical proof step by step and explain the reasoning",
      false,
    );

    expect(result.type).toBe("reasoning");
    expect(result.complexity).toBe("complex");
    expect(result.requiresReasoning).toBe(true);
  });

  it("marks vision requirements when images are provided", () => {
    const result = classifyTask("What is in this image?", true);

    expect(result.type).toBe("vision");
    expect(result.requiresVision).toBe(true);
  });

  it("uses hasImages to force vision requirements for non-visual prompts", () => {
    const result = classifyTask("Write a function", true);

    expect(result.requiresVision).toBe(true);
  });

  it("respects hint overrides for task type and complexity", () => {
    const result = classifyTask("Hello", false, {
      taskType: "reasoning",
      complexity: "complex",
    });

    expect(result.type).toBe("reasoning");
    expect(result.complexity).toBe("complex");
    expect(result.requiresReasoning).toBe(true);
    expect(result.requiresVision).toBe(false);
  });
});
