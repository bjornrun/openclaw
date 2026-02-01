import { type OpenClawConfig, loadConfig } from "../config/config.js";
import type { ModelCapabilities, ModelScore } from "../config/types.models.js";
import { resolveOpenClawAgentDir } from "./agent-paths.js";
import { inferModelCapabilities, type ModelCapabilityDefaults } from "./model-capabilities.js";
import { ensureOpenClawModelsJson } from "./models-config.js";

type ModelCostConfig = {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
};

export type ModelCatalogEntry = {
  id: string;
  name: string;
  provider: string;
  contextWindow?: number;
  reasoning?: boolean;
  input?: Array<"text" | "image">;
  cost?: ModelCostConfig;
  capabilities?: ModelCapabilities;
};

type DiscoveredModel = {
  id: string;
  name?: string;
  provider: string;
  contextWindow?: number;
  reasoning?: boolean;
  input?: Array<"text" | "image">;
  cost?: ModelCostConfig;
  capabilities?: ModelCapabilities;
};

type PiSdkModule = typeof import("./pi-model-discovery.js");

let modelCatalogPromise: Promise<ModelCatalogEntry[]> | null = null;
let hasLoggedModelCatalogError = false;
const defaultImportPiSdk = () => import("./pi-model-discovery.js");
let importPiSdk = defaultImportPiSdk;

export function resetModelCatalogCacheForTest() {
  modelCatalogPromise = null;
  hasLoggedModelCatalogError = false;
  importPiSdk = defaultImportPiSdk;
}

// Test-only escape hatch: allow mocking the dynamic import to simulate transient failures.
export function __setModelCatalogImportForTest(loader?: () => Promise<PiSdkModule>) {
  importPiSdk = loader ?? defaultImportPiSdk;
}

export async function loadModelCatalog(params?: {
  config?: OpenClawConfig;
  useCache?: boolean;
}): Promise<ModelCatalogEntry[]> {
  if (params?.useCache === false) {
    modelCatalogPromise = null;
  }
  if (modelCatalogPromise) {
    return modelCatalogPromise;
  }

  modelCatalogPromise = (async () => {
    const models: ModelCatalogEntry[] = [];
    const sortModels = (entries: ModelCatalogEntry[]) =>
      entries.sort((a, b) => {
        const p = a.provider.localeCompare(b.provider);
        if (p !== 0) {
          return p;
        }
        return a.name.localeCompare(b.name);
      });
    try {
      const cfg = params?.config ?? loadConfig();
      await ensureOpenClawModelsJson(cfg);
      // IMPORTANT: keep the dynamic import *inside* the try/catch.
      // If this fails once (e.g. during a pnpm install that temporarily swaps node_modules),
      // we must not poison the cache with a rejected promise (otherwise all channel handlers
      // will keep failing until restart).
      const piSdk = await importPiSdk();
      const agentDir = resolveOpenClawAgentDir();
      const { join } = await import("node:path");
      const authStorage = new piSdk.AuthStorage(join(agentDir, "auth.json"));
      const registry = new piSdk.ModelRegistry(authStorage, join(agentDir, "models.json")) as
        | {
            getAll: () => Array<DiscoveredModel>;
          }
        | Array<DiscoveredModel>;
      const entries = Array.isArray(registry) ? registry : registry.getAll();
      for (const entry of entries) {
        const id = String(entry?.id ?? "").trim();
        if (!id) {
          continue;
        }
        const provider = String(entry?.provider ?? "").trim();
        if (!provider) {
          continue;
        }
        const name = String(entry?.name ?? id).trim() || id;
        const contextWindow =
          typeof entry?.contextWindow === "number" && entry.contextWindow > 0
            ? entry.contextWindow
            : undefined;
        const reasoning = typeof entry?.reasoning === "boolean" ? entry.reasoning : undefined;
        const input = Array.isArray(entry?.input) ? entry.input : undefined;
        const cost = resolveModelCost(entry?.cost);
        const capabilities = resolveModelCapabilities(entry?.capabilities);
        models.push({ id, name, provider, contextWindow, reasoning, input, cost, capabilities });
      }

      if (models.length === 0) {
        // If we found nothing, don't cache this result so we can try again.
        modelCatalogPromise = null;
      }

      return sortModels(models);
    } catch (error) {
      if (!hasLoggedModelCatalogError) {
        hasLoggedModelCatalogError = true;
        console.warn(`[model-catalog] Failed to load model catalog: ${String(error)}`);
      }
      // Don't poison the cache on transient dependency/filesystem issues.
      modelCatalogPromise = null;
      if (models.length > 0) {
        return sortModels(models);
      }
      return [];
    }
  })();

  return modelCatalogPromise;
}

/**
 * Check if a model supports image input based on its catalog entry.
 */
export function modelSupportsVision(entry: ModelCatalogEntry | undefined): boolean {
  if (typeof entry?.capabilities?.supportsVision === "boolean") {
    return entry.capabilities.supportsVision;
  }
  return entry?.input?.includes("image") ?? false;
}

export function modelCatalogEntryToCapabilities(
  entry: ModelCatalogEntry,
  defaults?: ModelCapabilityDefaults,
): ModelCapabilities {
  return inferModelCapabilities(entry, defaults);
}

export type ModelRefWithScore = Pick<ModelScore, "provider" | "model"> & {
  score?: ModelScore;
};

/**
 * Find a model in the catalog by provider and model ID.
 */
export function findModelInCatalog(
  catalog: ModelCatalogEntry[],
  provider: string,
  modelId: string,
): ModelCatalogEntry | undefined {
  const normalizedProvider = provider.toLowerCase().trim();
  const normalizedModelId = modelId.toLowerCase().trim();
  return catalog.find(
    (entry) =>
      entry.provider.toLowerCase() === normalizedProvider &&
      entry.id.toLowerCase() === normalizedModelId,
  );
}

function resolveModelCost(value: unknown): ModelCostConfig | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const cost = value as ModelCostConfig;
  const input =
    typeof cost.input === "number" && Number.isFinite(cost.input) ? cost.input : undefined;
  const output =
    typeof cost.output === "number" && Number.isFinite(cost.output) ? cost.output : undefined;
  const cacheRead =
    typeof cost.cacheRead === "number" && Number.isFinite(cost.cacheRead)
      ? cost.cacheRead
      : undefined;
  const cacheWrite =
    typeof cost.cacheWrite === "number" && Number.isFinite(cost.cacheWrite)
      ? cost.cacheWrite
      : undefined;
  if (
    input === undefined &&
    output === undefined &&
    cacheRead === undefined &&
    cacheWrite === undefined
  ) {
    return undefined;
  }
  return { input, output, cacheRead, cacheWrite };
}

function resolveModelCapabilities(value: unknown): ModelCapabilities | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  return value as ModelCapabilities;
}
