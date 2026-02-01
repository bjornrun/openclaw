export type ModelApi =
  | "openai-completions"
  | "openai-responses"
  | "anthropic-messages"
  | "google-generative-ai"
  | "github-copilot"
  | "bedrock-converse-stream";

export type ModelCompatConfig = {
  supportsStore?: boolean;
  supportsDeveloperRole?: boolean;
  supportsReasoningEffort?: boolean;
  maxTokensField?: "max_completion_tokens" | "max_tokens";
};

export type ModelProviderAuthMode = "api-key" | "aws-sdk" | "oauth" | "token";

export type ModelDefinitionConfig = {
  id: string;
  name: string;
  api?: ModelApi;
  reasoning: boolean;
  input: Array<"text" | "image">;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;
  maxTokens: number;
  headers?: Record<string, string>;
  compat?: ModelCompatConfig;
  capabilities?: ModelCapabilities;
};

export type ModelProviderConfig = {
  baseUrl: string;
  apiKey?: string;
  auth?: ModelProviderAuthMode;
  api?: ModelApi;
  headers?: Record<string, string>;
  authHeader?: boolean;
  models: ModelDefinitionConfig[];
};

export type BedrockDiscoveryConfig = {
  enabled?: boolean;
  region?: string;
  providerFilter?: string[];
  refreshInterval?: number;
  defaultContextWindow?: number;
  defaultMaxTokens?: number;
};

export type ModelsConfig = {
  mode?: "merge" | "replace";
  providers?: Record<string, ModelProviderConfig>;
  bedrockDiscovery?: BedrockDiscoveryConfig;
};

/** Cost tier for model selection scoring. */
export type ModelCostTier = "free" | "low" | "medium" | "high";

/** Task types for model routing (mirrors TaskType from task-classifier). */
export type ModelTaskType = "coding" | "reasoning" | "chat" | "vision" | "analysis" | "general";

/** Task complexity levels for model routing. */
export type ModelTaskComplexity = "simple" | "moderate" | "complex";

/** Model capabilities for task-based routing. */
export type ModelCapabilities = {
  /** Task types this model is suitable for. */
  taskTypes: ModelTaskType[];
  /** Highest complexity level this model can handle well. */
  maxComplexity: ModelTaskComplexity;
  /** Whether the model supports vision/image input. */
  supportsVision: boolean;
  /** Whether the model has reasoning capabilities. */
  supportsReasoning: boolean;
  /** Context window size in tokens. */
  contextWindow: number;
  /** Cost tier derived from input/output costs. */
  costTier: ModelCostTier;
};

/** Weights for model scoring during selection. */
export type ModelScoringWeights = {
  /** Weight for capability match (0-1, default: 0.4). */
  capabilityMatch: number;
  /** Weight for cost efficiency (0-1, default: 0.3). */
  costEfficiency: number;
  /** Weight for performance (0-1, default: 0.2). */
  performance: number;
  /** Weight for availability (0-1, default: 0.1). */
  availability: number;
};

/** Score breakdown for a model. */
export type ModelScoreBreakdown = {
  /** Capability match score (0-1). */
  capability: number;
  /** Cost efficiency score (0-1). */
  cost: number;
  /** Performance score (0-1). */
  performance: number;
  /** Availability score (0-1). */
  availability: number;
};

/** Complete score for a model candidate. */
export type ModelScore = {
  /** Provider name. */
  provider: string;
  /** Model ID. */
  model: string;
  /** Total weighted score (0-1). */
  totalScore: number;
  /** Individual score components. */
  breakdown: ModelScoreBreakdown;
  /** Human-readable explanation of the score. */
  reasoning: string;
};
