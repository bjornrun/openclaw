# Task-Based Model Routing

Task-based routing automatically selects the most appropriate model for each request based on task classification, model capabilities, and configured preferences.

## Overview

When enabled, the routing system:

1. Classifies incoming prompts by task type and complexity
2. Matches tasks to model capabilities
3. Scores available models based on configurable weights
4. Selects the optimal model while respecting user preferences

## When to Use Local Models (Ollama)

Local models are preferred for:

- **Simple chat conversations**: Greetings, acknowledgments, simple Q&A
- **Basic coding tasks**: Simple functions, code formatting, basic debugging
- **General text processing**: Summarization of short texts, simple rewrites
- **Small context requirements**: Tasks requiring < 32K tokens
- **Privacy-sensitive tasks**: Data stays local
- **High-frequency, low-complexity tasks**: Cost optimization

## When to Use Cloud Models

Cloud models are preferred for:

- **Complex reasoning tasks**: Multi-step logic, mathematical proofs, deep analysis
- **Vision tasks requiring high accuracy**: Unless a capable local vision model is available
- **Large context requirements**: Tasks requiring > 100K tokens
- **Latest model capabilities**: Cutting-edge reasoning, specialized knowledge
- **Production-critical tasks**: High reliability requirements
- **Explicit model requests**: User-specified cloud models

## Routing Decision Flow

```text
1. Classify task from prompt and context
2. Check for manual model override (user-specified model takes precedence)
3. Apply task-specific routing rules if configured
4. Score available models based on task requirements
5. Filter by capability requirements (vision, reasoning, context window)
6. Apply strategy preference:
   - cost-optimized → prefer local
   - performance-optimized → prefer cloud
   - balanced → weighted scoring
7. Select highest-scoring model
8. Fall back to configured default if routing fails
```

## Task Types

| Type | Description | Example Prompts |
|------|-------------|-----------------|
| `coding` | Code-related tasks | "Debug this function", "Implement a class" |
| `reasoning` | Logic and analysis | "Prove this theorem", "Analyze the trade-offs" |
| `chat` | Conversational | "Hello", "Thanks for your help" |
| `vision` | Image processing | "What's in this screenshot?", "Describe the diagram" |
| `analysis` | Document analysis | "Summarize this report", "Compare these options" |
| `general` | Fallback category | Everything else |

## Complexity Levels

| Level | Indicators |
|-------|------------|
| `simple` | Short prompts, single operations, basic tasks |
| `moderate` | Medium-length prompts, standard complexity |
| `complex` | Long prompts, multi-step operations, architecture decisions |

## Cost Optimization Strategy

| Tier | Cost Level | Usage |
|------|------------|-------|
| 0 | Free | Ollama local models - prioritize for simple tasks |
| 1 | Low | Budget cloud models - use for moderate complexity |
| 2 | Medium | Standard cloud models - use for complex tasks |
| 3 | High | Premium models - reserve for critical/specialized tasks |

## Configuration

### Basic Configuration

```yaml
agents:
  defaults:
    routing:
      enabled: true
      strategy: balanced
      preferLocal: true
```

### Cost-Optimized (Ollama-First)

```yaml
agents:
  defaults:
    routing:
      enabled: true
      strategy: cost-optimized
      preferLocal: true
      localProviders: [ollama]
      taskRules:
        - taskType: coding
          complexity: simple
          preferredProviders: [ollama]
        - taskType: reasoning
          complexity: complex
          preferredProviders: [anthropic, openai]
        - taskType: vision
          preferredModels: [ollama/llama-vision, anthropic/claude-opus-4.5]
```

### Performance-Optimized

```yaml
agents:
  defaults:
    routing:
      enabled: true
      strategy: performance-optimized
      preferLocal: false
      taskRules:
        - taskType: coding
          minContextWindow: 100000
          preferredProviders: [anthropic, openai]
        - taskType: reasoning
          requireReasoning: true
          preferredModels: [anthropic/claude-opus-4.5]
```

### Balanced with Custom Weights

```yaml
agents:
  defaults:
    routing:
      enabled: true
      strategy: balanced
      scoringWeights:
        capabilityMatch: 0.5
        costEfficiency: 0.2
        performance: 0.2
        availability: 0.1
      taskRules:
        - taskType: chat
          preferredProviders: [ollama]
        - taskType: vision
          excludeProviders: [ollama]
```

## Configuration Reference

### ModelRoutingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable task-based routing |
| `strategy` | string | `balanced` | `cost-optimized`, `performance-optimized`, or `balanced` |
| `preferLocal` | boolean | `true` | Prefer local models when suitable |
| `localProviders` | string[] | `[ollama]` | Providers considered local |
| `cloudProviders` | string[] | `[anthropic, openai, google]` | Providers considered cloud |
| `taskRules` | TaskRoutingRule[] | - | Per-task routing overrides |
| `scoringWeights` | ModelScoringWeights | - | Custom scoring weights |
| `fallbackBehavior` | string | `default-model` | `manual-selection` or `default-model` |

### TaskRoutingRule

| Field | Type | Description |
|-------|------|-------------|
| `taskType` | string | Task type this rule applies to |
| `complexity` | string | Optional complexity filter |
| `preferredProviders` | string[] | Preferred providers for this task |
| `preferredModels` | string[] | Specific model refs (provider/model) |
| `excludeProviders` | string[] | Providers to exclude |
| `minContextWindow` | number | Minimum context window required |
| `requireReasoning` | boolean | Whether reasoning capability is required |

### ModelScoringWeights

| Field | Default | Description |
|-------|---------|-------------|
| `capabilityMatch` | 0.4 | Weight for capability match |
| `costEfficiency` | 0.3 | Weight for cost efficiency |
| `performance` | 0.2 | Weight for performance |
| `availability` | 0.1 | Weight for availability |

## Backward Compatibility

- Routing is **disabled by default** (`enabled: false`)
- Existing configurations work without changes
- Manual model selection always takes precedence
- Falls back to configured default model if routing fails

## Ollama Model Detection

The routing system automatically detects Ollama model capabilities from model names:

| Capability | Detection Patterns |
|------------|-------------------|
| Vision | `vision`, `vlm`, `llava`, `bakllava`, `moondream` |
| Coding | `code`, `coder`, `codellama`, `starcoder`, `deepseek-coder`, `qwen-coder` |
| Reasoning | `r1`, `reasoning`, `deepseek-r1`, `qwen-r1` |
| Context Window | Extracted from name (e.g., `32k`, `128k`) |
