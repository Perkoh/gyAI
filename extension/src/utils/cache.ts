/**
 * gyAI — Local Result Cache (extension/src/utils/cache.ts)
 * ========================================================
 *
 * A thin, TTL-aware wrapper over `chrome.storage.local` (blueprint §12.1 step 4:
 * "check chrome.storage.local for a cached result (1-hour TTL)").
 *
 * Storage model: one entry per domain under a namespaced key,
 * `adis:cache:<domain>` — mirroring the server-side Redis key (§4.2). Per-domain
 * keys keep reads/writes granular and avoid the read-modify-write races a single
 * blob would introduce when the service worker analyzes several tabs at once.
 *
 * Privacy (§12.2): keys are bare domain names — no URLs, paths, or identifiers.
 * `clearCache()` removes every gyAI entry (exposed via the popup Settings).
 *
 * Robustness: caching is a non-critical optimization, so every storage error is
 * swallowed and treated as a miss / no-op rather than propagated to the UI. All
 * functions accept an injectable `now` for deterministic testing.
 *
 * Cross-browser: uses the callback form of the storage API (universally
 * supported by Chrome MV3 and Firefox's `chrome` namespace) wrapped in Promises,
 * and no-ops gracefully if `chrome.storage` is unavailable.
 *
 * Depends only on ./types.
 */

import { AnalyzeResponse, CachedAnalysis, DEFAULT_CACHE_TTL_MS } from './types';

/** Namespace prefix for all cache keys (mirrors the server Redis key, §4.2). */
export const CACHE_PREFIX = 'adis:cache:';

/** Build the chrome.storage key for a domain. */
export function cacheKey(domain: string): string {
  return `${CACHE_PREFIX}${domain.trim().toLowerCase()}`;
}

/** Resolve the local storage area, or `null` if the API isn't available. */
function localArea(): chrome.storage.StorageArea | null {
  if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
    return chrome.storage.local;
  }
  return null;
}

function storageGet(keys: string | string[] | null): Promise<Record<string, unknown>> {
  const area = localArea();
  if (!area) return Promise.resolve({});
  return new Promise((resolve, reject) => {
    area.get(keys, (items) => {
      const err = chrome.runtime?.lastError;
      if (err) reject(new Error(err.message ?? 'storage.get failed'));
      else resolve((items ?? {}) as Record<string, unknown>);
    });
  });
}

function storageSet(items: Record<string, unknown>): Promise<void> {
  const area = localArea();
  if (!area) return Promise.resolve();
  return new Promise((resolve, reject) => {
    area.set(items, () => {
      const err = chrome.runtime?.lastError;
      if (err) reject(new Error(err.message ?? 'storage.set failed'));
      else resolve();
    });
  });
}

function storageRemove(keys: string | string[]): Promise<void> {
  const area = localArea();
  if (!area) return Promise.resolve();
  return new Promise((resolve, reject) => {
    area.remove(keys, () => {
      const err = chrome.runtime?.lastError;
      if (err) reject(new Error(err.message ?? 'storage.remove failed'));
      else resolve();
    });
  });
}

/**
 * Return the full cached entry for a domain (result + timestamps), or `null` on
 * a miss or expiry. Expired entries are opportunistically removed.
 */
export async function getCachedEntry(
  domain: string,
  now: number = Date.now(),
): Promise<CachedAnalysis | null> {
  const key = cacheKey(domain);
  let items: Record<string, unknown>;
  try {
    items = await storageGet([key]);
  } catch {
    return null; // storage error → treat as a miss
  }

  const entry = items[key] as CachedAnalysis | undefined;
  if (!entry || typeof entry.expiresAt !== 'number') return null;

  if (entry.expiresAt <= now) {
    void invalidate(domain); // fire-and-forget cleanup of the stale entry
    return null;
  }
  return entry;
}

/** Return just the cached {@link AnalyzeResponse} for a domain, or `null`. */
export async function getCachedResult(
  domain: string,
  now: number = Date.now(),
): Promise<AnalyzeResponse | null> {
  const entry = await getCachedEntry(domain, now);
  return entry ? entry.result : null;
}

/**
 * Cache an analysis result for a domain.
 *
 * `ttlMs` defaults to the blueprint's 1-hour local TTL (§12.1). The service
 * worker may pass a shorter TTL for flagged domains if it wants them to
 * re-check sooner — the store itself is TTL-agnostic.
 */
export async function setCachedResult(
  domain: string,
  result: AnalyzeResponse,
  ttlMs: number = DEFAULT_CACHE_TTL_MS,
  now: number = Date.now(),
): Promise<void> {
  const entry: CachedAnalysis = { result, cachedAt: now, expiresAt: now + ttlMs };
  try {
    await storageSet({ [cacheKey(domain)]: entry });
  } catch {
    // Ignore quota/storage errors — caching is best-effort.
  }
}

/** Remove the cached entry for a single domain. */
export async function invalidate(domain: string): Promise<void> {
  try {
    await storageRemove(cacheKey(domain));
  } catch {
    // ignore
  }
}

/**
 * Remove every ADIS cache entry (leaves settings and other keys intact).
 * Returns the number of entries cleared. Wired to the popup's "clear cache".
 */
export async function clearCache(): Promise<number> {
  let all: Record<string, unknown>;
  try {
    all = await storageGet(null);
  } catch {
    return 0;
  }
  const keys = Object.keys(all).filter((k) => k.startsWith(CACHE_PREFIX));
  if (keys.length === 0) return 0;
  try {
    await storageRemove(keys);
  } catch {
    return 0;
  }
  return keys.length;
}

/**
 * Sweep out expired (or malformed) cache entries. Optional housekeeping the
 * service worker can call periodically to keep storage bounded. Returns the
 * number of entries removed.
 */
export async function pruneExpired(now: number = Date.now()): Promise<number> {
  let all: Record<string, unknown>;
  try {
    all = await storageGet(null);
  } catch {
    return 0;
  }
  const expired = Object.keys(all).filter((k) => {
    if (!k.startsWith(CACHE_PREFIX)) return false;
    const entry = all[k] as CachedAnalysis | undefined;
    return !entry || typeof entry.expiresAt !== 'number' || entry.expiresAt <= now;
  });
  if (expired.length === 0) return 0;
  try {
    await storageRemove(expired);
  } catch {
    return 0;
  }
  return expired.length;
}