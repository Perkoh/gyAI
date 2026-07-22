/**
 * gyAI — Background Service Worker (extension/src/background/service_worker.ts)
 * ============================================================================
 *
 * The extension's brain (blueprint §12.1). Manifest V3 service worker that:
 *
 *   1. Listens for top-level navigations (`webNavigation`, main frame only) and
 *      extracts *only* the hostname — path/query are discarded for privacy
 *      (§12.2).
 *   2. Resolves a verdict for that domain: local cache first (`cache.ts`,
 *      1-hour TTL, §12.1 step 4), then the gyAI API (`api.ts`) on a miss.
 *   3. Applies the score decision tree (§4.3 / §11.1) via `tierForScore`:
 *        • score < 0.50  → silent, nothing injected.
 *        • 0.50–0.79     → caution (yellow) banner.
 *        • ≥ 0.80        → alert (red) banner.
 *      For flagged domains it injects the content script and hands it the
 *      result to render (§12.1 steps 8–9).
 *   4. Brokers runtime messages for the popup and content script (§11.3):
 *      current-tab status, feedback submission, cache clearing, and the
 *      "Leave This Site" action.
 *   5. Reflects per-tab status on the toolbar badge.
 *
 * User autonomy first (§1.1): this worker never blocks or redirects a page on
 * its own. It only informs. The single navigation it ever performs is an
 * explicit, user-initiated "Leave This Site" click relayed from the banner.
 *
 * Lifecycle note: MV3 workers are ephemeral and may be torn down between events.
 * All durable state lives in `chrome.storage.local` (via `cache.ts` and the
 * settings key); the in-memory `tabRecords` map is a best-effort optimization
 * that is safe to lose.
 *
 * Depends on ../utils/{types,api,cache}.
 */

import {
  AnalysisResultMessage,
  AnalyzeResponse,
  CurrentStatusMessage,
  DEFAULT_SETTINGS,
  ExtensionSettings,
  FeedbackAck,
  MessageType,
  NotificationTier,
  RuntimeMessage,
  SubmitFeedbackMessage,
  tierForScore,
} from '../utils/types';
import { AdisApiClient } from '../utils/api';
import {
  clearCache,
  getCachedResult,
  pruneExpired,
  setCachedResult,
} from '../utils/cache';

/* ============================================================================
 * Constants
 * ========================================================================== */

/**
 * chrome.storage key holding {@link ExtensionSettings}. Deliberately NOT under
 * the `adis:cache:` prefix so `clearCache()` (§ popup Settings) leaves settings
 * untouched.
 */
const SETTINGS_KEY = 'adis:settings';

/**
 * Bundled path of the content script, relative to the extension root. Must
 * match the filename Webpack emits (step 27 / build item #29). Adjust here if
 * the bundle lands elsewhere.
 */
const CONTENT_SCRIPT_FILE = 'content/content_script.js';

/**
 * Where a tab is sent when the user clicks "Leave This Site" in an alert banner
 * (§11.2). A neutral blank page — we navigate away from the risky site without
 * choosing a destination for the user.
 */
const LEAVE_SITE_URL = 'about:blank';

/**
 * Shorter local TTL for flagged (suspicious/malicious) domains so the extension
 * re-checks them sooner than safe ones — mirroring the server's tiered Redis
 * TTL (§4.1: 1 hr safe / 15 min flagged). Safe domains use `settings.cacheTtlMs`.
 */
const FLAGGED_CACHE_TTL_MS = 15 * 60 * 1000;

/** Per-tier toolbar badge appearance. `silent` clears the badge. */
const BADGE: Record<NotificationTier, { text: string; color: string }> = {
  silent: { text: '', color: '#00000000' },
  caution: { text: '!', color: '#F9A825' }, // amber
  alert: { text: '!', color: '#D32F2F' }, // red
};

/* ============================================================================
 * In-memory per-tab state (best-effort; see lifecycle note above)
 * ========================================================================== */

interface TabRecord {
  /** The committed URL this record was created for (guards against races). */
  url: string;
  /** Hostname extracted from `url`. */
  domain: string;
  /** In-flight (or settled) analysis for this navigation. */
  analysis: Promise<AnalyzeResponse | null>;
}

const tabRecords = new Map<number, TabRecord>();

/**
 * De-duplicates concurrent analyses of the same domain across tabs, so opening
 * five tabs on one host issues at most one API call (supports the "local cache
 * prevents duplicate API calls" acceptance test, §Phase 5).
 */
const inFlight = new Map<string, Promise<AnalyzeResponse | null>>();

/* ============================================================================
 * Settings
 * ========================================================================== */

/** Promise wrapper over `chrome.storage.local.get` (callback form → Firefox-safe). */
function storageGet(keys: string | string[] | null): Promise<Record<string, unknown>> {
  const area = chrome.storage?.local;
  if (!area) return Promise.resolve({});
  return new Promise((resolve) => {
    area.get(keys, (items) => {
      // Reads never reject here — a storage error simply yields defaults.
      void chrome.runtime?.lastError;
      resolve((items ?? {}) as Record<string, unknown>);
    });
  });
}

/** Promise wrapper over `chrome.storage.local.set`. */
function storageSet(items: Record<string, unknown>): Promise<void> {
  const area = chrome.storage?.local;
  if (!area) return Promise.resolve();
  return new Promise((resolve) => {
    area.set(items, () => {
      void chrome.runtime?.lastError;
      resolve();
    });
  });
}

/**
 * Load settings, merged over {@link DEFAULT_SETTINGS} so a stored partial (e.g.
 * from a prior extension version) never leaves a field undefined.
 */
async function getSettings(): Promise<ExtensionSettings> {
  const items = await storageGet(SETTINGS_KEY);
  const stored = items[SETTINGS_KEY] as Partial<ExtensionSettings> | undefined;
  return { ...DEFAULT_SETTINGS, ...(stored ?? {}) };
}

/** Persist any missing default keys (idempotent; run on install/update). */
async function ensureSettingsPersisted(): Promise<void> {
  const merged = await getSettings();
  await storageSet({ [SETTINGS_KEY]: merged });
}

/* ============================================================================
 * API client (rebuilt only when the configured base URL changes)
 * ========================================================================== */

let cachedClient: AdisApiClient | null = null;
let cachedBaseUrl = '';

function apiClientFor(settings: ExtensionSettings): AdisApiClient {
  if (!cachedClient || cachedBaseUrl !== settings.apiBaseUrl) {
    cachedClient = new AdisApiClient({ baseUrl: settings.apiBaseUrl });
    cachedBaseUrl = settings.apiBaseUrl;
  }
  return cachedClient;
}

/* ============================================================================
 * Domain extraction (privacy boundary — §12.2)
 * ========================================================================== */

/**
 * Extract the bare, normalized hostname to analyze, or `null` if the URL should
 * be skipped entirely (non-web scheme, browser-internal page, or a local host
 * for which a reputation lookup is meaningless).
 *
 * This is the ONLY value that ever leaves the browser (§12.2): no scheme, path,
 * query, or fragment.
 */
function domainFromUrl(rawUrl: string): string | null {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;

  const host = url.hostname.toLowerCase().replace(/\.$/, '');
  if (!host) return null;

  // Skip hosts where analysis adds nothing (and avoid noisy API traffic).
  if (
    host === 'localhost' ||
    host.endsWith('.local') ||
    host.endsWith('.localhost') ||
    host === '127.0.0.1' ||
    host === '::1' ||
    host.startsWith('127.')
  ) {
    return null;
  }
  return host;
}

/* ============================================================================
 * Core analysis: cache → API → cache
 * ========================================================================== */

/**
 * Resolve a verdict for `domain`. Returns the {@link AnalyzeResponse} on
 * success, or `null` when the domain is unanalyzable / the API is unreachable
 * (analysis is best-effort and must never surface an error to the page).
 */
async function analyzeDomain(
  domain: string,
  settings: ExtensionSettings,
): Promise<AnalyzeResponse | null> {
  // 1) Local cache (§12.1 step 4–5).
  const cached = await getCachedResult(domain);
  if (cached) return cached;

  // 2) Collapse concurrent lookups of the same domain into one request.
  const pending = inFlight.get(domain);
  if (pending) return pending;

  // 3) Miss → hit the API, then store (§12.1 steps 6–7).
  const task = (async (): Promise<AnalyzeResponse | null> => {
    const result = await apiClientFor(settings).analyze(domain);
    if (!result.ok) {
      // Silent, non-fatal: log and treat as "unknown" (no banner shown).
      // eslint-disable-next-line no-console
      console.warn(`[ADIS] analyze(${domain}) failed: ${(result as any).error.code} — ${(result as any).error.message}`);
      return null;
    }
    const flagged = tierForScore(result.data.score) !== 'silent';
    const ttl = flagged ? FLAGGED_CACHE_TTL_MS : settings.cacheTtlMs;
    await setCachedResult(domain, result.data, ttl);
    return result.data;
  })().finally(() => inFlight.delete(domain));

  inFlight.set(domain, task);
  return task;
}

/* ============================================================================
 * Banner delivery (content script injection — §12.1 steps 8–9)
 * ========================================================================== */

/**
 * Send the scored result to the tab's content script, injecting the script
 * first if it isn't already present. The send-first/inject-on-failure order
 * makes this safe whether the content script is already loaded (SPA re-render)
 * or not (fresh navigation), and avoids double-registering listeners.
 */
async function deliverBanner(
  tabId: number,
  result: AnalyzeResponse,
  tier: Exclude<NotificationTier, 'silent'>,
): Promise<void> {
  const message: AnalysisResultMessage = {
    type: MessageType.AnalysisResult,
    result,
    tier,
  };
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch {
    // No receiver yet → inject the content script, then retry once.
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: [CONTENT_SCRIPT_FILE],
      });
      await chrome.tabs.sendMessage(tabId, message);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[ADIS] could not deliver banner to tab ${tabId}:`, err);
    }
  }
}

/**
 * Decide whether a settled result should surface a banner in `tabId`, honoring
 * the user's "caution banners" preference, and render it if so.
 */
async function maybeShowBanner(
  tabId: number,
  result: AnalyzeResponse | null,
  settings: ExtensionSettings,
): Promise<void> {
  if (!result) return;
  const tier = tierForScore(result.score);
  if (tier === 'silent') return;
  if (tier === 'caution' && !settings.showCautionBanners) return;
  await deliverBanner(tabId, result, tier);
}

/* ============================================================================
 * Toolbar badge
 * ========================================================================== */

function setBadge(tabId: number, tier: NotificationTier): void {
  const cfg = BADGE[tier];
  void chrome.action.setBadgeBackgroundColor({ tabId, color: cfg.color });
  void chrome.action.setBadgeText({ tabId, text: cfg.text });
}

function clearBadge(tabId: number): void {
  void chrome.action.setBadgeText({ tabId, text: '' });
}

/* ============================================================================
 * Navigation listeners (§12.1 steps 1–9)
 *
 * Two-phase design: analysis is kicked off on `onCommitted` (as early as the
 * navigation is real), but the banner is injected on `onCompleted` when a DOM
 * exists to attach to. The analysis promise is stored on the tab record so the
 * completion handler simply awaits whatever the commit handler started.
 * ========================================================================== */

chrome.webNavigation.onCommitted.addListener(async (details) => {
  if (details.frameId !== 0) return; // main frame only (§12.1 step 2)

  const settings = await getSettings();
  if (!settings.enabled) {
    // Master switch off (§1.1): stay completely silent and unmarked.
    tabRecords.delete(details.tabId);
    clearBadge(details.tabId);
    return;
  }

  const domain = domainFromUrl(details.url);
  if (!domain) {
    tabRecords.delete(details.tabId);
    clearBadge(details.tabId);
    return;
  }

  const analysis = analyzeDomain(domain, settings).then((result) => {
    // Only mutate UI if this navigation is still the tab's current one.
    if (tabRecords.get(details.tabId)?.url === details.url) {
      setBadge(details.tabId, result ? tierForScore(result.score) : 'silent');
    }
    return result;
  });

  tabRecords.set(details.tabId, { url: details.url, domain, analysis });
});

chrome.webNavigation.onCompleted.addListener(async (details) => {
  if (details.frameId !== 0) return;

  const record = tabRecords.get(details.tabId);
  if (!record || record.url !== details.url) return; // superseded by a newer nav

  const settings = await getSettings();
  if (!settings.enabled) return;

  const result = await record.analysis;
  await maybeShowBanner(details.tabId, result, settings);
});

/**
 * Single-page-app route changes don't fire commit/complete, so re-evaluate here
 * when the hostname actually changes. Same-host client-side navigations are
 * ignored — the existing verdict for the host still stands.
 */
chrome.webNavigation.onHistoryStateUpdated.addListener(async (details) => {
  if (details.frameId !== 0) return;

  const domain = domainFromUrl(details.url);
  const prev = tabRecords.get(details.tabId);
  if (!domain || prev?.domain === domain) return;

  const settings = await getSettings();
  if (!settings.enabled) return;

  const analysis = analyzeDomain(domain, settings);
  tabRecords.set(details.tabId, { url: details.url, domain, analysis });

  const result = await analysis;
  if (tabRecords.get(details.tabId)?.url !== details.url) return;
  setBadge(details.tabId, result ? tierForScore(result.score) : 'silent');
  await maybeShowBanner(details.tabId, result, settings);
});

/* ============================================================================
 * Tab lifecycle
 * ========================================================================== */

/** Keep the badge in sync when the user switches tabs. */
chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  const domain = tab?.url ? domainFromUrl(tab.url) : null;
  if (!domain) {
    clearBadge(tabId);
    return;
  }
  const result = await getCachedResult(domain);
  setBadge(tabId, result ? tierForScore(result.score) : 'silent');
});

/** Drop in-memory state when a tab closes. */
chrome.tabs.onRemoved.addListener((tabId) => {
  tabRecords.delete(tabId);
});

/* ============================================================================
 * Runtime messaging (popup ↔ background ↔ content — §8 of types.ts, §11.3)
 * ========================================================================== */

chrome.runtime.onMessage.addListener(
  (message: RuntimeMessage, sender, sendResponse) => {
    switch (message.type) {
      case MessageType.GetCurrentStatus:
        handleGetCurrentStatus().then(sendResponse);
        return true; // async response

      case MessageType.SubmitFeedback:
        handleSubmitFeedback(message).then(sendResponse);
        return true; // async response

      case MessageType.ClearCache:
        clearCache()
          .then(() => sendResponse({ success: true }))
          .catch(() => sendResponse({ success: false }));
        return true; // async response

      case MessageType.LeaveSite:
        void handleLeaveSite(sender);
        return false; // fire-and-forget

      default:
        return false;
    }
  },
);

/** Popup → background: report the active tab's domain + latest known verdict. */
async function handleGetCurrentStatus(): Promise<CurrentStatusMessage> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const domain = tab?.url ? domainFromUrl(tab.url) : null;
  const result = domain ? await getCachedResult(domain) : null;
  return { type: MessageType.CurrentStatus, domain, result };
}

/** Popup/content → background: relay a feedback report to the API (§10.1). */
async function handleSubmitFeedback(
  message: SubmitFeedbackMessage,
): Promise<FeedbackAck> {
  const settings = await getSettings();
  const result = await apiClientFor(settings).submitFeedback(message.feedback);
  return result.ok ? result.data : { success: false, message: (result as any).error.message };
}

/**
 * Content → background: the user clicked "Leave This Site" (§11.2). This is the
 * one and only user-initiated navigation ADIS ever performs — consistent with
 * "user autonomy first" (§1.1).
 */
async function handleLeaveSite(sender: chrome.runtime.MessageSender): Promise<void> {
  const tabId = sender.tab?.id;
  if (tabId == null) return;
  try {
    await chrome.tabs.update(tabId, { url: LEAVE_SITE_URL });
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(`[ADIS] leave-site navigation failed for tab ${tabId}:`, err);
  }
}

/* ============================================================================
 * Install / startup
 * ========================================================================== */

chrome.runtime.onInstalled.addListener(async () => {
  await ensureSettingsPersisted(); // seed DEFAULT_SETTINGS on first install
  void pruneExpired(); // opportunistic cache housekeeping
});

chrome.runtime.onStartup.addListener(() => {
  void pruneExpired();
});