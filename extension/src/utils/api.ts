/**
 * gyAI — API Client (extension/src/utils/api.ts)
 * ==============================================
 *
 * Thin, typed `fetch` wrapper around the gyAI Flask API (blueprint §10). Every
 * method resolves to an {@link ApiResult} discriminated union — success or a
 * structured {@link ApiError} — so callers branch on `result.ok` instead of
 * wrapping calls in try/catch (network failures and timeouts are surfaced as
 * errors, never thrown).
 *
 * Design points tied to the blueprint:
 *   • Base URL defaults to the production API (§10.1); overridable so the
 *     service worker can pass `settings.apiBaseUrl`.
 *   • Privacy (§12.2): only the bare domain is sent; cookies are never attached
 *     (`credentials: 'omit'`), and no identifiers are added.
 *   • HTTPS is required for extensions (§FLAG 5) — enforce it via the base URL.
 *   • Rate limiting (§10.3): a 429 is mapped to a `RATE_LIMITED` error for the
 *     caller to back off on; the client never auto-retries (avoids hammering a
 *     rate limit and prevents duplicate feedback writes).
 *   • Every request has a hard client-side timeout via `AbortController`.
 *
 * Depends only on ./types.
 */

import {
  AnalyzeRequest,
  AnalyzeResponse,
  AnalyzeResult,
  ApiError,
  ApiResult,
  ErrorCode,
  FeedbackAck,
  FeedbackRequest,
  HealthResponse,
  VersionResponse,
  DEFAULT_API_BASE_URL,
  isAnalyzeResponse,
  isApiErrorResponse,
} from './types';

/** Default per-request timeout (ms). Analysis is meant to feel instant. */
export const DEFAULT_REQUEST_TIMEOUT_MS = 8_000;

/** Options accepted by {@link AdisApiClient}. */
export interface AdisApiClientOptions {
  /** API base URL, e.g. `https://adis-api.fly.dev/api/v1`. Must be HTTPS. */
  baseUrl?: string;
  /** Per-request timeout in milliseconds. */
  timeoutMs?: number;
  /** Optional `X-API-Key` for authenticated endpoints (not needed for /analyze). */
  apiKey?: string;
}

/** Build a failed {@link ApiResult} from parts. */
function failure(code: ErrorCode, message: string, status: number): { ok: false; error: ApiError } {
  return { ok: false, error: { code, message, status } };
}

/** Parse JSON without throwing; returns `null` on malformed input. */
function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * Client for the ADIS REST API. Instantiate once (or use the shared
 * {@link adisApi} default) and reuse across calls.
 */
export class AdisApiClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly apiKey?: string;

  constructor(options: AdisApiClientOptions = {}) {
    // Normalize: strip trailing slashes so `${baseUrl}/analyze` is well-formed.
    this.baseUrl = (options.baseUrl ?? DEFAULT_API_BASE_URL).replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    this.apiKey = options.apiKey;

    if (!/^https:\/\//i.test(this.baseUrl)) {
      // Extensions can only call HTTPS endpoints (§FLAG 5). Warn loudly rather
      // than fail construction, so misconfiguration is obvious in dev.
      // eslint-disable-next-line no-console
      console.warn(`[ADIS] API base URL is not HTTPS: ${this.baseUrl}`);
    }
  }

  /**
   * `POST /analyze` — score a single domain (§10.2).
   * The domain is normalized (trimmed, lower-cased, trailing dot removed).
   * Only the domain leaves the browser (§12.2).
   */
  async analyze(domain: string): Promise<AnalyzeResult> {
    const normalized = AdisApiClient.normalizeDomain(domain);
    if (!normalized) {
      return failure('INVALID_DOMAIN', 'Empty or invalid domain supplied to analyze().', 422);
    }
    const payload: AnalyzeRequest = { domain: normalized };
    return this.request<AnalyzeResponse>(
      '/analyze',
      { method: 'POST', headers: this.headers(true), body: JSON.stringify(payload) },
      isAnalyzeResponse,
    );
  }

  /**
   * `POST /feedback` — submit a false-positive/confirmation report (§9, §10.1).
   * A 2xx with an empty body is treated as `{ success: true }`.
   */
   
  async submitFeedback(feedback: FeedbackRequest): Promise<ApiResult<FeedbackAck>> {
    const result = await this.request<FeedbackAck>('/feedback', {
      method: 'POST',
      headers: this.headers(true),
      body: JSON.stringify(feedback),
    });
    if (result.ok) {
      // Any 2xx means the server accepted the feedback (201 "received" or
      // 202 "accepted"). The server body has no `success` field — it signals
      // success via HTTP status — so normalize to the FeedbackAck shape here,
      // preserving the server's message when present.
      const data = result.data as { message?: string } | null;
      return { ok: true, data: { success: true, message: data?.message } };
    }
    return result;
  }

  /** `GET /version` — current model version info (§10.1). */
  async getVersion(): Promise<ApiResult<VersionResponse>> {
    return this.request<VersionResponse>('/version', {
      method: 'GET',
      headers: this.headers(false),
    });
  }

  /** `GET /health` — liveness check for monitoring / popup status (§10.1). */
  async getHealth(): Promise<ApiResult<HealthResponse>> {
    return this.request<HealthResponse>('/health', {
      method: 'GET',
      headers: this.headers(false),
    });
  }

  // ------------------------------------------------------------------ internals

  /** Assemble request headers; `json` toggles the Content-Type. */
  private headers(json: boolean): Record<string, string> {
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (json) headers['Content-Type'] = 'application/json';
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;
    return headers;
  }

  /**
   * Perform a request and normalize the outcome into an {@link ApiResult}.
   * Never throws: timeouts, offline errors, HTTP errors and malformed bodies
   * are all mapped to structured errors.
   */
  private async request<T>(
    path: string,
    init: RequestInit,
    validate?: (body: unknown) => body is T,
  ): Promise<ApiResult<T>> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
        credentials: 'omit', // never send cookies (§12.2)
        cache: 'no-store', // the extension owns caching (cache.ts)
      });

      const raw = await response.text();
      const body: unknown = raw ? safeJsonParse(raw) : null;

      if (!response.ok) {
        // Prefer the server's structured error envelope when present (§10.2).
        if (isApiErrorResponse(body)) {
          const err = body.error;
          return { ok: false, error: { ...err, status: err.status || response.status } };
        }
        const code: ErrorCode = response.status === 429 ? 'RATE_LIMITED' : 'HTTP_ERROR';
        return failure(code, `ADIS API returned HTTP ${response.status}.`, response.status);
      }

      if (validate && !validate(body)) {
        return failure('BAD_RESPONSE', `Malformed response body from ${path}.`, response.status);
      }

      return { ok: true, data: body as T };
    } catch (err) {
      const name = (err as { name?: string })?.name;
      if (name === 'AbortError') {
        return failure('TIMEOUT', `Request to ${path} timed out after ${this.timeoutMs} ms.`, 0);
      }
      const message = err instanceof Error ? err.message : String(err);
      return failure('NETWORK_ERROR', `Could not reach the ADIS API: ${message}`, 0);
    } finally {
      clearTimeout(timer);
    }
  }

  /** Trim, lower-case, and drop a single trailing dot from a hostname. */
  private static normalizeDomain(domain: string): string {
    return domain.trim().toLowerCase().replace(/\.$/, '');
  }
}

/**
 * Shared client using {@link DEFAULT_API_BASE_URL}. Convenient for simple call
 * sites; construct your own {@link AdisApiClient} when you need a custom base
 * URL (e.g. from user settings) or API key.
 */
export const adisApi = new AdisApiClient();
