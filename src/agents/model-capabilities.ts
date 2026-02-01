import type {
  ModelCapabilities,
  ModelCostTier,
  ModelTaskComplexity,
  ModelTaskType,
} from "../config/types.models.js";

export type ModelCapabilityInput = {
  id: string;
  name?: string;
  reasoning?: boolean;
  input?: Array<"text" | "image">;
  contextWindow?: number;
  cost?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  capabilities?: ModelCapabilities;
};

export type ModelCapabilityDefaults = {
  defaultContextWindow?: number;
  defaultCostTier?: ModelCostTier;
};

const DEFAULT_CONTEXT_WINDOW = 8192;

export const MODEL_VISION_PATTERNS = /vision|vlm|llava|bakllava|moondream/i;
export const MODEL_CODING_PATTERNS = /code|coder|codellama|starcoder|deepseek-coder|qwen.*coder/i;
export const MODEL_REASONING_PATTERNS = /r1|reasoning|deepseek-r1|qwen-r1/i;
export const MODEL_CONTEXT_PATTERNS = /(\d+)k/i;

export function detectModelNameCapabilities(modelId: string): {
  supportsVision: boolean;
  supportsCoding: boolean;
  supportsReasoning: boolean;
  contextWindow?: number;
} {
  const lowerName = modelId.toLowerCase();

  const supportsVision = MODEL_VISION_PATTERNS.test(lowerName);
  const supportsCoding = MODEL_CODING_PATTERNS.test(lowerName);
  const supportsReasoning = MODEL_REASONING_PATTERNS.test(lowerName);

  let contextWindow: number | undefined;
  const contextMatch = MODEL_CONTEXT_PATTERNS.exec(lowerName);
  if (contextMatch?.[1]) {
    const parsed = parseInt(contextMatch[1], 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      contextWindow = parsed * 1000;
    }
  }

  return { supportsVision, supportsCoding, supportsReasoning, contextWindow };
}

function inferCostTier(cost: ModelCapabilityInput["cost"], fallback: ModelCostTier): ModelCostTier {
  if (!cost) {
    return fallback;
  }
  const input = typeof cost.input === "number" && Number.isFinite(cost.input) ? cost.input : 0;
  const output = typeof cost.output === "number" && Number.isFinite(cost.output) ? cost.output : 0;
  const cacheRead =
    typeof cost.cacheRead === "number" && Number.isFinite(cost.cacheRead) ? cost.cacheRead : 0;
  const cacheWrite =
    typeof cost.cacheWrite === "number" && Number.isFinite(cost.cacheWrite) ? cost.cacheWrite : 0;
  const total = input + output + cacheRead + cacheWrite;
  if (total === 0) {
    return "free";
  }
  if (total <= 2) {
    return "low";
  }
  if (total <= 10) {
    return "medium";
  }
  return "high";
}

function inferMaxComplexity(
  contextWindow: number,
  supportsReasoning: boolean,
): ModelTaskComplexity {
  if (supportsReasoning || contextWindow >= 100_000) {
    return "complex";
  }
  if (contextWindow >= 32_000) {
    return "moderate";
  }
  return "simple";
}

function inferTaskTypes(params: {
  supportsVision: boolean;
  supportsReasoning: boolean;
  supportsCoding: boolean;
  contextWindow: number;
}): ModelTaskType[] {
  const taskTypes = new Set<ModelTaskType>(["general", "chat"]);
  if (params.supportsVision) {
    taskTypes.add("vision");
  }
  if (params.supportsReasoning) {
    taskTypes.add("reasoning");
  }
  if (params.supportsCoding) {
    taskTypes.add("coding");
  }
  if (params.supportsReasoning || params.contextWindow >= 32_000) {
    taskTypes.add("analysis");
  }
  return Array.from(taskTypes);
}

export function inferModelCapabilities(
  model: ModelCapabilityInput,
  defaults?: ModelCapabilityDefaults,
): ModelCapabilities {
  if (model.capabilities) {
    return model.capabilities;
  }
  const name = model.name?.trim();
  const detectionTarget = name ? `${name} ${model.id}` : model.id;
  const nameSignals = detectModelNameCapabilities(detectionTarget);

  const supportsVision = Array.isArray(model.input)
    ? model.input.includes("image")
    : nameSignals.supportsVision;
  const supportsReasoning =
    typeof model.reasoning === "boolean" ? model.reasoning : nameSignals.supportsReasoning;
  const supportsCoding = nameSignals.supportsCoding;
  const contextWindow =
    (typeof model.contextWindow === "number" && model.contextWindow > 0
      ? model.contextWindow
      : nameSignals.contextWindow) ??
    defaults?.defaultContextWindow ??
    DEFAULT_CONTEXT_WINDOW;

  return {
    taskTypes: inferTaskTypes({
      supportsVision,
      supportsReasoning,
      supportsCoding,
      contextWindow,
    }),
    maxComplexity: inferMaxComplexity(contextWindow, supportsReasoning),
    supportsVision,
    supportsReasoning,
    contextWindow,
    costTier: inferCostTier(model.cost, defaults?.defaultCostTier ?? "medium"),
  };
}
