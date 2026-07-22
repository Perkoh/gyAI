/**
 * gyAI — Browser Extension Shared Types
 * =====================================
 *
 * Single source of truth for every type crossing a module boundary inside the
 * extension. This file is the root of the extension's dependency graph: it
 * imports nothing and is imported by `api.ts`, `cache.ts`, the service worker,
 * the content script, and the popup.
 *
 * Two naming conventions are used deliberately:
 *   • **Wire types** (`AnalyzeRequest`, `AnalyzeResponse`, `ApiError`,
 *     `FeedbackRequest`, `VersionResponse`) mirror the JSON exchanged with the
 *     Flask API *exactly*, so their fields stay `snake_case` (blueprint §10.2).
 *     Do not rename these fields — they must match the server contract.
 *   • **Internal types** (cache entries, settings, runtime messages) are local
 *     to the extension and use idiomatic `camelCase`.
 *
 * Blueprint references are cited inline (e.g. "§10.2") against
 * PROJECT_BLUEPRINT.md v2.0.
 */

/* ============================================================================
 * 1. Core enumerations (single-source-of-truth pattern)
 *
 * Each enumeration is declared once as a readonly tuple and the corresponding
 * union type is derived from it. This gives both a runtime array (handy for
 * iterating options in the popup / validating input) and a compile-time union.
 * ========================================================================== */

/** Classification returned by the model (§4.3, §10.2). */
export const RISK_LABELS = ['safe', 'suspicious', 'malicious'] as const;
export type RiskLabel = (typeof RISK_LABELS)[number];

/** Model confidence band (§10.2; Supabase CHECK constraint §9). */
export const CONFIDENCE_LEVELS = ['low', 'medium', 'high'] as const;
export type ConfidenceLevel = (typeof CONFIDENCE_LEVELS)[number];

/**
 * Notification tier the extension renders, derived from the score (§4.3, §11.1):
 *   • `silent`  — score < 0.50 → nothing shown.
 *   • `caution` — 0.50 ≤ score < 0.80 → yellow banner, auto-dismiss after 8s.
 *   • `alert`   — score ≥ 0.80 → red banner, stays until dismissed.
 */
export const NOTIFICATION_TIERS = ['silent', 'caution', 'alert'] as const;
export type NotificationTier = (typeof NOTIFICATION_TIERS)[number];

/** User verdict when reporting a domain (§9 `user_feedback.user_verdict`). */
export const FEEDBACK_VERDICTS = [
  'false_positive',
  'confirmed_malicious',
  'unsure',
] as const;
export type FeedbackVerdict = (typeof FEEDBACK_VERDICTS)[number];

/* ============================================================================
 * 2. Analyze endpoint — request & response (§10.2)
 * ========================================================================== */

/**
 * Request body for `POST /api/v1/analyze`.
 *
 * Privacy (§12.2): only the bare hostname is ever sent — never the full URL,
 * path, query string, cookies, or any user identifier.
 */
export interface AnalyzeRequest {
  /** Hostname only, e.g. `"secure-login-paypa1.xyz"` (no scheme, no path). */
  domain: string;
}

/**
 * Successful response from `POST /api/v1/analyze`.
 *
 * Field shape matches the server JSON one-for-one (§10.2). `reasons` is empty
 * and `analysis_id` is `null` for safe domains (only suspicious/malicious
 * results are logged and assigned an id, §1.2 / §9).
 */
export interface AnalyzeResponse {
  /** Echo of the analyzed domain. */
  domain: string;
  /** Malicious probability in the range 0..1 (e.g. `0.9341`). */
  score: number;
  /** Bucketed classification derived from `score`. */
  label: RiskLabel;
  /** Model confidence band for this prediction. */
  confidence: ConfidenceLevel;
  /** Human-readable reasons (top SHAP contributors). Empty when safe. */
  reasons: string[];
  /** Version string of the model that produced this result, e.g. `"v1.2.0"`. */
  model_version: string;
  /** UUID of the logged analysis; `null` for safe (unlogged) domains. */
  analysis_id: string | null;
  /** Server-side analysis time in milliseconds. */
  duration_ms: number;
  /** `true` if served from the server-side Redis cache. */
  cached: boolean;
  /** `true` if live DNS/WHOIS features were available for this analysis. */
  network_features_used: boolean;
}

/** Response from `GET /api/v1/version` (§10.1). Extra fields tolerated. */
export interface VersionResponse {
  model_version: string;
  [key: string]: unknown;
}

/** Response from `GET /api/v1/health` (§10.1). */
export interface HealthResponse {
  /** e.g. `"ok"` | `"healthy"`. */
  status: string;
  [key: string]: unknown;
}

/* ============================================================================
 * 3. Error contract (§10.2)
 * ========================================================================== */

/**
 * Known error codes returned by the API. Kept as a widened `string` union so
 * unforeseen codes still type-check, while documenting the ones the blueprint
 * implies (INVALID_DOMAIN is the only code shown explicitly, §10.2; the others
 * follow from the 400/422/429/500 handlers in the build order).
 */
export type ErrorCode =
  | 'INVALID_DOMAIN'
  | 'VALIDATION_ERROR'
  | 'RATE_LIMITED'
  | 'INTERNAL_ERROR'
  // eslint-disable-next-line @typescript-eslint/ban-types
  | (string & {});

/** The inner error object (§10.2). */
export interface ApiError {
  code: ErrorCode;
  message: string;
  /** HTTP status mirrored in the body, e.g. `422`. */
  status: number;
}

/** Full error envelope: `{ "error": { ... } }` (§10.2). */
export interface ApiErrorResponse {
  error: ApiError;
}

/* ============================================================================
 * 4. Client-side result wrapper
 *
 * `api.ts` (step 24) should resolve to this discriminated union instead of
 * throwing, so callers can branch on `ok` without try/catch. A network failure
 * (offline, timeout) is surfaced as an `ApiError` with a synthetic code.
 * ========================================================================== */

export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: ApiError };

/** Convenience alias for the analyze call's result. */
export type AnalyzeResult = ApiResult<AnalyzeResponse>;

/* ============================================================================
 * 5. Feedback (§9 `user_feedback`, `POST /api/v1/feedback`)
 * ========================================================================== */

/** Request body for `POST /api/v1/feedback`. Fields map to the DB columns. */
export interface FeedbackRequest {
  domain: string;
  /** What ADIS predicted (the `label` shown to the user). */
  system_label: RiskLabel;
  /** The user's judgement. */
  user_verdict: FeedbackVerdict;
  /** Optional free-text note. */
  user_comment?: string;
}

/** Acknowledgement returned after submitting feedback. */
export interface FeedbackAck {
  success: boolean;
  message?: string;
}

/* ============================================================================
 * 6. Local cache (chrome.storage.local wrapper — cache.ts, step 25)
 *
 * §12.1 step 4: results are cached locally with a 1-hour TTL keyed by domain.
 * These are internal storage records, hence camelCase.
 * ========================================================================== */

/** One cached analysis plus the metadata needed to expire it. */
export interface CachedAnalysis {
  /** The API result being cached. */
  result: AnalyzeResponse;
  /** Epoch milliseconds when this entry was stored. */
  cachedAt: number;
  /** Epoch milliseconds after which this entry is considered stale. */
  expiresAt: number;
}

/** Shape of the extension's cache namespace inside chrome.storage.local. */
export type DomainCache = Record<string, CachedAnalysis>;

/* ============================================================================
 * 7. Extension settings (popup Settings, §11.3)
 * ========================================================================== */

export interface ExtensionSettings {
  /** Master on/off switch — user autonomy first (§1.1). */
  enabled: boolean;
  /** Base URL of the ADIS API (must be HTTPS, §FLAG 5). */
  apiBaseUrl: string;
  /** Whether to show the yellow `caution` tier (some users only want alerts). */
  showCautionBanners: boolean;
  /** Local cache TTL in milliseconds. */
  cacheTtlMs: number;
}

/* ============================================================================
 * 8. Runtime messaging (service worker ↔ content script ↔ popup)
 *
 * Manifest V3 uses chrome.runtime / chrome.tabs message passing. Every message
 * is a discriminated union member keyed by `type`, so handlers can switch
 * exhaustively. Type strings are namespaced with `ADIS_` to avoid ambiguity.
 * ========================================================================== */

/**
 * Message type discriminants, as a plain `as const` object (portable across
 * every bundler — unlike `const enum`, which can break under `isolatedModules`
 * / esbuild / babel-loader). Use these values when *constructing* messages;
 * the interfaces below pin the same literals as their discriminant.
 */
export const MessageType = {
  /** Background → content script: render a banner for a scored domain. */
  AnalysisResult: 'ADIS_ANALYSIS_RESULT',
  /** Popup → background: request the active tab's current status. */
  GetCurrentStatus: 'ADIS_GET_CURRENT_STATUS',
  /** Background → popup: reply carrying the active tab's status. */
  CurrentStatus: 'ADIS_CURRENT_STATUS',
  /** Popup/content → background: submit a user feedback report. */
  SubmitFeedback: 'ADIS_SUBMIT_FEEDBACK',
  /** Popup → background: clear the local result cache (settings action). */
  ClearCache: 'ADIS_CLEAR_CACHE',
  /** Content → background: the user chose to leave a flagged site. */
  LeaveSite: 'ADIS_LEAVE_SITE',
} as const;
export type MessageType = (typeof MessageType)[keyof typeof MessageType];

/** Background → content script. Instructs the banner injection. */
export interface AnalysisResultMessage {
  type: 'ADIS_ANALYSIS_RESULT';
  result: AnalyzeResponse;
  /** Precomputed tier so the content script needn't re-derive thresholds. */
  tier: Exclude<NotificationTier, 'silent'>;
}

/** Popup → background. No payload. Response: {@link CurrentStatusMessage}. */
export interface GetCurrentStatusMessage {
  type: 'ADIS_GET_CURRENT_STATUS';
}

/** Background → popup. Current status of the active tab (§11.3). */
export interface CurrentStatusMessage {
  type: 'ADIS_CURRENT_STATUS';
  /** Active tab hostname, or `null` if none/unsupported (e.g. chrome://). */
  domain: string | null;
  /** Latest analysis for that domain, or `null` if not analyzed yet. */
  result: AnalyzeResponse | null;
}

/** Popup/content → background. Response: {@link FeedbackAck}. */
export interface SubmitFeedbackMessage {
  type: 'ADIS_SUBMIT_FEEDBACK';
  feedback: FeedbackRequest;
}

/** Popup → background. Clears chrome.storage.local domain cache. */
export interface ClearCacheMessage {
  type: 'ADIS_CLEAR_CACHE';
}

/** Content → background. The user clicked "Leave This Site". */
export interface LeaveSiteMessage {
  type: 'ADIS_LEAVE_SITE';
  domain: string;
}

/** Every message that can be sent within the extension. */
export type RuntimeMessage =
  | AnalysisResultMessage
  | GetCurrentStatusMessage
  | CurrentStatusMessage
  | SubmitFeedbackMessage
  | ClearCacheMessage
  | LeaveSiteMessage;

/**
 * Maps a request message type to the response type its sender should expect
 * from `chrome.runtime.sendMessage`. Useful for a typed messaging helper.
 */
export interface MessageResponseMap {
  ADIS_GET_CURRENT_STATUS: CurrentStatusMessage;
  ADIS_SUBMIT_FEEDBACK: FeedbackAck;
  ADIS_CLEAR_CACHE: { success: boolean };
}

/* ============================================================================
 * 9. Constants & pure helpers (decision-tree math, §4.3 / §11.1)
 *
 * There is no separate constants module in the extension's directory layout,
 * so the score thresholds and derivation helpers live here — the one place all
 * consumers already import. Keeping them centralized means the 0.50 / 0.80
 * boundaries and the 8-second auto-dismiss are defined exactly once.
 * ========================================================================== */

/** Score boundaries for the notification tiers (§4.3). */
export const RISK_THRESHOLDS = {
  /** Inclusive lower bound for the `caution`/`suspicious` tier. */
  suspicious: 0.5,
  /** Inclusive lower bound for the `alert`/`malicious` tier. */
  malicious: 0.8,
} as const;

/** Caution (yellow) banners auto-dismiss after this long (§11.1). */
export const CAUTION_AUTO_DISMISS_MS = 8_000;

/** Default local-cache TTL: 1 hour (§12.1). */
export const DEFAULT_CACHE_TTL_MS = 60 * 60 * 1000;

/** Default API base URL (§10.1). Must be HTTPS for the extension (§FLAG 5). */
export const DEFAULT_API_BASE_URL = 'https://gyai-api.fly.dev/api/v1';

/** Default settings applied on first install. */
export const DEFAULT_SETTINGS: ExtensionSettings = {
  enabled: true,
  apiBaseUrl: DEFAULT_API_BASE_URL,
  showCautionBanners: true,
  cacheTtlMs: DEFAULT_CACHE_TTL_MS,
};

/**
 * Derive the notification tier from a raw score (§4.3).
 * Kept as the single implementation of the decision tree so the service worker
 * and content script never re-hardcode the thresholds.
 */
export function tierForScore(score: number): NotificationTier {
  if (score >= RISK_THRESHOLDS.malicious) return 'alert';
  if (score >= RISK_THRESHOLDS.suspicious) return 'caution';
  return 'silent';
}

/** Derive the risk label from a raw score. Mirrors {@link tierForScore}. */
export function labelForScore(score: number): RiskLabel {
  if (score >= RISK_THRESHOLDS.malicious) return 'malicious';
  if (score >= RISK_THRESHOLDS.suspicious) return 'suspicious';
  return 'safe';
}

/** Whether a score warrants any banner at all (i.e. not silent). */
export function shouldNotify(score: number): boolean {
  return score >= RISK_THRESHOLDS.suspicious;
}

/* ============================================================================
 * 10. Narrowing helpers
 * ========================================================================== */

/** Runtime type guard: is a parsed API body the error envelope? */
export function isApiErrorResponse(body: unknown): body is ApiErrorResponse {
  return (
    typeof body === 'object' &&
    body !== null &&
    'error' in body &&
    typeof (body as ApiErrorResponse).error === 'object'
  );
}

/** Runtime type guard: does a value look like a well-formed AnalyzeResponse? */
export function isAnalyzeResponse(body: unknown): body is AnalyzeResponse {
  if (typeof body !== 'object' || body === null) return false;
  const b = body as Record<string, unknown>;
  return (
    typeof b.domain === 'string' &&
    typeof b.score === 'number' &&
    typeof b.label === 'string' &&
    Array.isArray(b.reasons)
  );
}