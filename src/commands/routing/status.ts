import type { AuthProfileStore } from "../../agents/auth-profiles.js";
import type { RuntimeEnv } from "../../runtime.js";
import type { TableColumn } from "../../terminal/table.js";
import { resolveOpenClawAgentDir } from "../../agents/agent-paths.js";
import { resolveAgentDir } from "../../agents/agent-scope.js";
import { ensureAuthProfileStore } from "../../agents/auth-profiles.js";
import { checkOllamaHealth } from "../../agents/ollama-health.js";
import { getGlobalRoutingMetrics } from "../../agents/routing-metrics.js";
import { loadRoutingConfig, mergeRoutingConfig } from "../../agents/task-router.js";
import { loadConfig } from "../../config/config.js";
import { renderTable } from "../../terminal/table.js";
import { theme } from "../../terminal/theme.js";
import { resolveKnownAgentId } from "../models/shared.js";

export type RoutingStatusOpts = {
  agent?: string;
  json?: boolean;
  plain?: boolean;
};

export async function routingStatusCommand(opts: RoutingStatusOpts, runtime: RuntimeEnv) {
  const cfg = loadConfig();
  const agentId = resolveKnownAgentId({ cfg, rawAgentId: opts.agent });
  const agentDir = agentId ? resolveAgentDir(cfg, agentId) : resolveOpenClawAgentDir();
  const routingConfig = loadRoutingConfig(cfg);
  const mergedConfig = mergeRoutingConfig(routingConfig);

  // Build status object
  const status = {
    enabled: mergedConfig.enabled,
    strategy: mergedConfig.strategy,
    preferLocal: mergedConfig.preferLocal,
    localProviders: mergedConfig.localProviders,
    cloudProviders: mergedConfig.cloudProviders,
    fallbackBehavior: mergedConfig.fallbackBehavior,
    ollama: {
      healthCheck: mergedConfig.ollama.healthCheck,
      healthCheckTimeoutMs: mergedConfig.ollama.healthCheckTimeoutMs,
      preferForTasks: mergedConfig.ollama.preferForTasks,
      excludeForTasks: mergedConfig.ollama.excludeForTasks,
      health: null as { available: boolean; latencyMs: number | null; modelCount: number } | null,
    },
    taskRules: mergedConfig.taskRules,
    scoringWeights: mergedConfig.scoringWeights,
    metrics: null as ReturnType<typeof getGlobalRoutingMetrics.prototype.getSnapshot> | null,
  };

  // Check Ollama health if enabled
  if (mergedConfig.ollama.healthCheck) {
    try {
      const authStore = ensureAuthProfileStore(agentDir, { allowKeychainPrompt: false });
      const authHeader = resolveOllamaAuthHeader(authStore);
      const healthResult = await checkOllamaHealth({
        timeoutMs: mergedConfig.ollama.healthCheckTimeoutMs,
        authHeader,
      });
      status.ollama.health = {
        available: healthResult.available,
        latencyMs: healthResult.latencyMs,
        modelCount: healthResult.modelCount,
      };
    } catch {
      status.ollama.health = { available: false, latencyMs: null, modelCount: 0 };
    }
  }

  // Get metrics snapshot
  try {
    const metrics = getGlobalRoutingMetrics();
    status.metrics = metrics.getSnapshot();
  } catch {
    // Metrics not available
  }

  if (opts.json) {
    runtime.log(JSON.stringify(status, null, 2));
    return;
  }

  if (opts.plain) {
    runtime.log(`enabled=${status.enabled}`);
    runtime.log(`strategy=${status.strategy}`);
    runtime.log(`preferLocal=${status.preferLocal}`);
    runtime.log(`localProviders=${status.localProviders.join(",")}`);
    runtime.log(`cloudProviders=${status.cloudProviders.join(",")}`);
    runtime.log(`fallbackBehavior=${status.fallbackBehavior}`);
    runtime.log(`ollama.healthCheck=${status.ollama.healthCheck}`);
    if (status.ollama.health) {
      runtime.log(`ollama.available=${status.ollama.health.available}`);
      runtime.log(`ollama.modelCount=${status.ollama.health.modelCount}`);
    }
    return;
  }

  // Rich output
  runtime.log(theme.heading("\n📊 Routing Status\n"));

  // Enabled state
  const enabledLabel = status.enabled ? theme.success("✓ Enabled") : theme.muted("✗ Disabled");
  runtime.log(`  ${theme.accent("Status:")} ${enabledLabel}`);
  runtime.log(`  ${theme.accent("Strategy:")} ${status.strategy}`);
  runtime.log(`  ${theme.accent("Prefer Local:")} ${status.preferLocal ? "yes" : "no"}`);
  runtime.log(`  ${theme.accent("Fallback:")} ${status.fallbackBehavior}`);
  runtime.log("");

  // Provider preferences
  runtime.log(theme.heading("Provider Preferences"));
  runtime.log(`  ${theme.accent("Local:")} ${status.localProviders.join(", ") || "(none)"}`);
  runtime.log(`  ${theme.accent("Cloud:")} ${status.cloudProviders.join(", ") || "(none)"}`);
  runtime.log("");

  // Ollama settings
  runtime.log(theme.heading("Ollama Configuration"));
  runtime.log(
    `  ${theme.accent("Health Check:")} ${status.ollama.healthCheck ? "enabled" : "disabled"}`,
  );
  runtime.log(`  ${theme.accent("Timeout:")} ${status.ollama.healthCheckTimeoutMs}ms`);
  runtime.log(
    `  ${theme.accent("Prefer For:")} ${status.ollama.preferForTasks.join(", ") || "(none)"}`,
  );
  runtime.log(
    `  ${theme.accent("Exclude For:")} ${status.ollama.excludeForTasks.join(", ") || "(none)"}`,
  );

  if (status.ollama.health) {
    const healthStatus = status.ollama.health.available
      ? theme.success(
          `✓ Available (${status.ollama.health.modelCount} models, ${status.ollama.health.latencyMs}ms)`,
        )
      : theme.error("✗ Unavailable");
    runtime.log(`  ${theme.accent("Health:")} ${healthStatus}`);
  }
  runtime.log("");

  // Scoring weights
  runtime.log(theme.heading("Scoring Weights"));
  runtime.log(
    `  ${theme.accent("Capability Match:")} ${(status.scoringWeights.capabilityMatch * 100).toFixed(0)}%`,
  );
  runtime.log(
    `  ${theme.accent("Cost Efficiency:")} ${(status.scoringWeights.costEfficiency * 100).toFixed(0)}%`,
  );
  runtime.log(
    `  ${theme.accent("Performance:")} ${(status.scoringWeights.performance * 100).toFixed(0)}%`,
  );
  runtime.log(
    `  ${theme.accent("Availability:")} ${(status.scoringWeights.availability * 100).toFixed(0)}%`,
  );
  runtime.log("");

  // Task rules
  if (status.taskRules.length > 0) {
    runtime.log(theme.heading("Task Rules"));
    const columns: TableColumn[] = [
      { key: "taskType", header: "Task Type", minWidth: 12 },
      { key: "complexity", header: "Complexity", minWidth: 10 },
      { key: "providers", header: "Preferred Providers", minWidth: 20, flex: true },
      { key: "requirements", header: "Requirements", minWidth: 15, flex: true },
    ];

    const rows = status.taskRules.map((rule) => ({
      taskType: rule.taskType,
      complexity: rule.complexity ?? "any",
      providers: rule.preferredProviders?.join(", ") ?? "-",
      requirements:
        [
          rule.requireReasoning ? "reasoning" : null,
          rule.minContextWindow ? `ctx≥${rule.minContextWindow}` : null,
        ]
          .filter(Boolean)
          .join(", ") || "-",
    }));

    runtime.log(renderTable({ columns, rows }));
  } else {
    runtime.log(theme.muted("  No task rules configured.\n"));
  }

  // Metrics summary
  if (status.metrics && status.metrics.totalDecisions > 0) {
    runtime.log(theme.heading("Metrics Summary"));
    runtime.log(`  ${theme.accent("Total Decisions:")} ${status.metrics.totalDecisions}`);
    runtime.log(`  ${theme.accent("Fallbacks Used:")} ${status.metrics.fallbacksUsed}`);
    runtime.log(`  ${theme.accent("Overrides Used:")} ${status.metrics.overridesUsed}`);
    runtime.log(`  ${theme.accent("Errors:")} ${status.metrics.errors}`);

    if (Object.keys(status.metrics.decisionsByTaskType).length > 0) {
      runtime.log(`  ${theme.accent("By Task Type:")}`);
      for (const [taskType, count] of Object.entries(status.metrics.decisionsByTaskType)) {
        runtime.log(`    ${taskType}: ${String(count)}`);
      }
    }
    runtime.log("");
  }
}

function resolveOllamaAuthHeader(store: AuthProfileStore): { Authorization: string } | null {
  const profiles = store?.profiles ?? {};
  const profileIds = Object.keys(profiles).filter((id) => id.startsWith("ollama:"));
  if (profileIds.length === 0) {
    return null;
  }
  const profile = profiles[profileIds[0]];
  if (profile?.type === "api_key" && profile.key) {
    return { Authorization: `Bearer ${profile.key}` };
  }
  if (profile?.type === "token" && profile.token) {
    return { Authorization: `Bearer ${profile.token}` };
  }
  return null;
}
