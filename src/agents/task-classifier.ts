/**
 * Task classification types and detection logic for intelligent model routing.
 */

/** Task type categories for model routing. */
export type TaskType = "coding" | "reasoning" | "chat" | "vision" | "analysis" | "general";

/** Task complexity levels. */
export type TaskComplexity = "simple" | "moderate" | "complex";

/** Result of classifying a user prompt. */
export interface TaskClassification {
  /** Primary task type detected. */
  type: TaskType;
  /** Estimated complexity of the task. */
  complexity: TaskComplexity;
  /** Whether the task requires vision/image processing. */
  requiresVision: boolean;
  /** Estimated context tokens needed (if determinable). */
  estimatedContextTokens?: number;
  /** Whether the task requires reasoning capabilities. */
  requiresReasoning: boolean;
  /** Confidence score for the classification (0-1). */
  confidence: number;
}

/** Manual hints for task classification from directives. */
export interface TaskClassificationHints {
  /** Override task type. */
  taskType?: TaskType;
  /** Override complexity. */
  complexity?: TaskComplexity;
  /** Force vision requirement. */
  requiresVision?: boolean;
  /** Force reasoning requirement. */
  requiresReasoning?: boolean;
}

/** Keyword patterns for task type detection. */
const TASK_PATTERNS: Record<TaskType, RegExp[]> = {
  coding: [
    /\b(debug|implement|refactor|function|class|method|variable|code|compile|syntax|bug|fix|error|exception)\b/i,
    /\b(typescript|javascript|python|rust|go|java|c\+\+|css|html|sql|api|endpoint)\b/i,
    /\b(test|unit test|integration|mock|stub|coverage)\b/i,
    /\b(git|commit|merge|branch|pull request|pr)\b/i,
  ],
  reasoning: [
    /\b(analyze|prove|deduce|calculate|solve|derive|infer|reason|logic|theorem)\b/i,
    /\b(step[- ]by[- ]step|think through|work out|figure out|explain why)\b/i,
    /\b(mathematical|equation|formula|proof|hypothesis)\b/i,
    /\b(compare and contrast|evaluate|assess|critique)\b/i,
  ],
  vision: [
    /\b(image|photo|picture|screenshot|diagram|chart|graph|visual|ui|design)\b/i,
    /\b(look at|see|view|show|display|render|draw)\b/i,
    /\b(ocr|recognize|identify|detect|scan)\b/i,
  ],
  analysis: [
    /\b(summarize|summary|overview|review|analyze|analysis|evaluate|assessment)\b/i,
    /\b(compare|contrast|difference|similarity|pros and cons)\b/i,
    /\b(report|findings|conclusion|recommendation)\b/i,
    /\b(data|metrics|statistics|trends|patterns)\b/i,
  ],
  chat: [
    /\b(hello|hi|hey|thanks|thank you|please|sorry|excuse me)\b/i,
    /\b(how are you|what's up|good morning|good evening)\b/i,
    /\b(yes|no|ok|okay|sure|got it|understood)\b/i,
  ],
  general: [],
};

/** Complexity indicators. */
const COMPLEXITY_PATTERNS = {
  complex: [
    /\b(complex|complicated|advanced|sophisticated|comprehensive|thorough|detailed)\b/i,
    /\b(multi[- ]step|multiple|several|many|all|entire|complete)\b/i,
    /\b(architecture|system|design|refactor|rewrite|overhaul)\b/i,
    /\b(optimize|performance|scale|production|enterprise)\b/i,
  ],
  simple: [
    /\b(simple|basic|quick|easy|small|minor|trivial|just|only)\b/i,
    /\b(one|single|a|the)\s+(function|method|line|file|change)\b/i,
    /\b(typo|rename|format|lint|style)\b/i,
  ],
};

/**
 * Classify a task based on the prompt content and context.
 *
 * @param prompt - The user's prompt text
 * @param hasImages - Whether the request includes images
 * @param hints - Optional manual classification hints
 * @returns Task classification result
 */
export function classifyTask(
  prompt: string,
  hasImages: boolean,
  hints?: TaskClassificationHints,
): TaskClassification {
  // Apply manual overrides if provided
  if (hints?.taskType) {
    return {
      type: hints.taskType,
      complexity: hints.complexity ?? inferComplexity(prompt),
      requiresVision: hints.requiresVision ?? hasImages,
      requiresReasoning: hints.requiresReasoning ?? hints.taskType === "reasoning",
      confidence: 1.0,
    };
  }

  // Detect task type from patterns
  const scores: Record<TaskType, number> = {
    coding: 0,
    reasoning: 0,
    chat: 0,
    vision: 0,
    analysis: 0,
    general: 0,
  };

  for (const [taskType, patterns] of Object.entries(TASK_PATTERNS) as [TaskType, RegExp[]][]) {
    for (const pattern of patterns) {
      if (pattern.test(prompt)) {
        scores[taskType] += 1;
      }
    }
  }

  // Vision gets a boost if images are present
  if (hasImages) {
    scores.vision += 3;
  }

  // Find the highest scoring task type
  let maxScore = 0;
  let detectedType: TaskType = "general";
  for (const [taskType, score] of Object.entries(scores) as [TaskType, number][]) {
    if (score > maxScore) {
      maxScore = score;
      detectedType = taskType;
    }
  }

  // Calculate confidence based on score differential
  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const confidence = totalScore > 0 ? Math.min(maxScore / totalScore + 0.3, 1.0) : 0.5;

  // Infer complexity
  const complexity = hints?.complexity ?? inferComplexity(prompt);

  // Determine if reasoning is required
  const requiresReasoning =
    hints?.requiresReasoning ??
    (detectedType === "reasoning" || complexity === "complex" || scores.reasoning > 1);

  return {
    type: detectedType,
    complexity,
    requiresVision: hints?.requiresVision ?? hasImages,
    requiresReasoning,
    confidence,
  };
}

/**
 * Infer task complexity from prompt content.
 */
function inferComplexity(prompt: string): TaskComplexity {
  // Check for complex indicators
  for (const pattern of COMPLEXITY_PATTERNS.complex) {
    if (pattern.test(prompt)) {
      return "complex";
    }
  }

  // Check for simple indicators
  for (const pattern of COMPLEXITY_PATTERNS.simple) {
    if (pattern.test(prompt)) {
      return "simple";
    }
  }

  // Use prompt length as a heuristic
  if (prompt.length < 100) {
    return "simple";
  }
  if (prompt.length > 500) {
    return "complex";
  }

  return "moderate";
}
