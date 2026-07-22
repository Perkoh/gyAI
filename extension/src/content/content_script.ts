/**
 * gyAI — Content Script (extension/src/content/content_script.ts)
 * ==============================================================
 *
 * Renders the non-intrusive security banner inside a visited page (blueprint
 * §11.2, §12.1 steps 9–10). This script is programmatically injected by the
 * service worker *only* on flagged domains (§12.1 step 8), then handed the
 * scored result via an {@link AnalysisResultMessage}.
 *
 * User autonomy first (§1.1): the banner never blocks or covers the page beyond
 * a dismissible top bar, and it never navigates on its own. The only navigation
 * it can trigger is an explicit "Leave This Site" click, which it relays to the
 * background worker (the worker owns all tab navigation).
 *
 * Tiers (§4.3 / §11.1):
 *   • caution (yellow) — top 2 reasons; auto-dismisses after 8s (pausable) or on
 *     user dismiss.
 *   • alert (red)      — top 3 reasons; stays until the user dismisses it.
 *
 * Isolation: the banner lives in an open Shadow DOM with `all: initial` on the
 * host, so page CSS can't distort it and its own CSS can't leak out. Styles come
 * from `content_script.css`, linked into the shadow root via
 * `chrome.runtime.getURL` (that file must be in `web_accessible_resources`).
 *
 * Idempotency: the service worker may re-inject this file (e.g. an SPA route
 * change). A per-frame guard ensures the message listener is registered once;
 * each new result simply replaces any banner already on screen.
 *
 * Depends only on ../utils/types (the worker brokers all API/cache access).
 */

import {
  AnalysisResultMessage,
  AnalyzeResponse,
  CAUTION_AUTO_DISMISS_MS,
  FeedbackAck,
  FeedbackRequest,
  LeaveSiteMessage,
  MessageType,
  NotificationTier,
  RuntimeMessage,
  SubmitFeedbackMessage,
} from '../utils/types';

declare global {
  interface Window {
    /** Set once per frame so re-injection doesn't double-register listeners. */
    __adisContentScriptLoaded?: boolean;
  }
}

/* ============================================================================
 * Constants & static copy (§11.2)
 * ========================================================================== */

/** id of the shadow host element we attach to the page. */
const HOST_ID = 'adis-security-banner-host';

/** How many reasons each tier shows (§4.3). */
const REASON_LIMIT: Record<Exclude<NotificationTier, 'silent'>, number> = {
  caution: 2,
  alert: 3,
};

/**
 * Per-tier presentation. Titles/summaries follow §11.2 verbatim; buttons follow
 * the §11.2 labels. `summary` takes the integer risk percentage.
 */
const TIER_COPY = {
  caution: {
    icon: '⚠',
    title: 'gyAI Security Notice',
    summary: (pct: number) => `This domain shows suspicious signals (Risk: ${pct}%)`,
    /** Left button (engage), right button (resolve). Neither leaves the page. */
    secondary: { action: 'learn-more' as const, label: 'Learn More' },
    primary: { action: 'dismiss' as const, label: "I'll Be Safe" },
    autoDismiss: true,
  },
  alert: {
    icon: '🚨',
    title: 'gyAI High Risk Warning',
    summary: (pct: number) =>
      `This domain is likely malicious or a phishing site (Risk: ${pct}%)`,
    secondary: { action: 'stay' as const, label: 'I Understand, Stay Anyway' },
    primary: { action: 'leave' as const, label: 'Leave This Site' },
    autoDismiss: false,
  },
} as const;

/** Learn-more body — reinforces that ADIS advises, never blocks (§1.1). */
const LEARN_MORE_TEXT =
  'gyAI flags domains with an automated model. These are risk signals, not ' +
  'proof of harm, and gyAI never blocks a site — the choice to continue is ' +
  'yours. If you believe this is a mistake, you can report it below.';

/* ============================================================================
 * Messaging helper (callback form → Chrome MV3 + Firefox compatible)
 * ========================================================================== */

function sendMessage<T = unknown>(message: RuntimeMessage): Promise<T | undefined> {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(message, (response) => {
        // Swallow "receiving end does not exist" and similar — non-critical.
        void chrome.runtime?.lastError;
        resolve(response as T);
      });
    } catch {
      resolve(undefined);
    }
  });
}

/* ============================================================================
 * Small DOM builder (textContent only — never innerHTML with server data)
 * ========================================================================== */

interface ElOptions {
  className?: string;
  text?: string;
  attrs?: Record<string, string>;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  opts: ElOptions = {},
  children: Node[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (opts.className) node.className = opts.className;
  if (opts.text != null) node.textContent = opts.text; // safe: no HTML parsing
  if (opts.attrs) {
    for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

/* ============================================================================
 * Banner lifecycle state (single banner at a time)
 * ========================================================================== */

let currentHost: HTMLElement | null = null;
let escapeHandler: ((e: KeyboardEvent) => void) | null = null;
let autoDismissTimer: number | null = null;

/** Tear down the current banner and any listeners/timers it owns. */
function removeBanner(): void {
  if (autoDismissTimer != null) {
    window.clearTimeout(autoDismissTimer);
    autoDismissTimer = null;
  }
  if (escapeHandler) {
    document.removeEventListener('keydown', escapeHandler, true);
    escapeHandler = null;
  }
  if (currentHost) {
    currentHost.remove();
    currentHost = null;
  }
}

/* ============================================================================
 * Auto-dismiss (caution only) — pausable on hover/focus (WCAG 2.2.1)
 * ========================================================================== */

function startAutoDismiss(banner: HTMLElement, ms: number): void {
  let remaining = ms;
  let startedAt = Date.now();

  const arm = () => {
    startedAt = Date.now();
    autoDismissTimer = window.setTimeout(removeBanner, Math.max(remaining, 0));
  };
  const pause = () => {
    if (autoDismissTimer != null) {
      window.clearTimeout(autoDismissTimer);
      autoDismissTimer = null;
      remaining -= Date.now() - startedAt;
    }
  };
  // Don't yank the banner away while the user is reading or interacting.
  banner.addEventListener('mouseenter', pause);
  banner.addEventListener('mouseleave', arm);
  banner.addEventListener('focusin', pause);
  banner.addEventListener('focusout', arm);

  arm();
}

/* ============================================================================
 * User actions
 * ========================================================================== */

/** Relay the explicit "Leave This Site" choice to the worker (§1.1, §11.2). */
function requestLeave(domain: string): void {
  const message: LeaveSiteMessage = { type: MessageType.LeaveSite, domain };
  void sendMessage(message);
}

/** Submit a false-positive report and confirm inline (§3, §10.1). */
async function reportFalsePositive(
  result: AnalyzeResponse,
  trigger: HTMLButtonElement,
): Promise<void> {
  trigger.disabled = true;
  trigger.textContent = 'Reporting…';

  const feedback: FeedbackRequest = {
    domain: result.domain,
    system_label: result.label,
    user_verdict: 'false_positive',
  };
  const message: SubmitFeedbackMessage = {
    type: MessageType.SubmitFeedback,
    feedback,
  };
  const ack = await sendMessage<FeedbackAck>(message);

  trigger.textContent = ack?.success ? 'Thanks — reported.' : 'Report failed, try again';
  trigger.disabled = Boolean(ack?.success);
}

/* ============================================================================
 * Banner construction
 * ========================================================================== */

function buildBanner(
  result: AnalyzeResponse,
  tier: Exclude<NotificationTier, 'silent'>,
): HTMLElement {
  const copy = TIER_COPY[tier];
  const pct = Math.round(result.score * 100);
  const reasons = result.reasons.slice(0, REASON_LIMIT[tier]);

  // Container — role announces urgency to assistive tech (alert vs status).
  const banner = el('section', {
    className: `adis-banner adis-banner--${tier}`,
    attrs: {
      role: tier === 'alert' ? 'alert' : 'status',
      'aria-label': `${copy.title}. ${copy.summary(pct)}`,
      tabindex: '-1',
    },
  });

  // Header: icon + title + close.
  const closeBtn = el('button', {
    className: 'adis-banner__close',
    text: '✕',
    attrs: { type: 'button', 'aria-label': 'Dismiss this gyAI notice' },
  });
  closeBtn.addEventListener('click', removeBanner);

  const header = el('div', { className: 'adis-banner__header' }, [
    el('span', { className: 'adis-banner__icon', text: copy.icon, attrs: { 'aria-hidden': 'true' } }),
    el('span', { className: 'adis-banner__title', text: copy.title }),
    closeBtn,
  ]);

  // Summary line with the risk percentage.
  const summary = el('p', { className: 'adis-banner__summary', text: copy.summary(pct) });

  // Reasons list (server-provided strings, inserted as text).
  const reasonList = el('ul', { className: 'adis-banner__reasons' });
  for (const reason of reasons) {
    reasonList.appendChild(el('li', { text: reason }));
  }

  // Collapsible "learn more" details (created up front, revealed on demand).
  const details = el('p', {
    className: 'adis-banner__details',
    text: LEARN_MORE_TEXT,
    attrs: { hidden: '' },
  });

  // Actions row.
  const actions = el('div', { className: 'adis-banner__actions' });

  const secondaryBtn = el('button', {
    className: 'adis-banner__btn adis-banner__btn--ghost',
    text: copy.secondary.label,
    attrs: { type: 'button' },
  });
  secondaryBtn.addEventListener('click', () => {
    if (copy.secondary.action === 'learn-more') {
      const isHidden = details.hasAttribute('hidden');
      if (isHidden) details.removeAttribute('hidden');
      else details.setAttribute('hidden', '');
      secondaryBtn.setAttribute('aria-expanded', String(isHidden));
    } else {
      // 'stay' — acknowledge and dismiss.
      removeBanner();
    }
  });
  if (copy.secondary.action === 'learn-more') {
    secondaryBtn.setAttribute('aria-expanded', 'false');
  }

  const primaryBtn = el('button', {
    className: 'adis-banner__btn adis-banner__btn--primary',
    text: copy.primary.label,
    attrs: { type: 'button' },
  });
  primaryBtn.addEventListener('click', () => {
    if (copy.primary.action === 'leave') requestLeave(result.domain);
    else removeBanner(); // 'dismiss' (I'll Be Safe)
  });

  actions.appendChild(secondaryBtn);
  actions.appendChild(primaryBtn);

  // Feedback affordance — lets users flag a false positive (§3 "collects feedback").
  const reportBtn = el('button', {
    className: 'adis-banner__report',
    text: 'Not a threat? Report it',
    attrs: { type: 'button' },
  });
  reportBtn.addEventListener('click', () => void reportFalsePositive(result, reportBtn));

  const footer = el('div', { className: 'adis-banner__footer' }, [reportBtn]);

  banner.appendChild(header);
  banner.appendChild(summary);
  if (reasons.length > 0) banner.appendChild(reasonList);
  banner.appendChild(details);
  banner.appendChild(actions);
  banner.appendChild(footer);

  return banner;
}

/* ============================================================================
 * Render (mount into an isolated shadow root)
 * ========================================================================== */

function renderBanner(
  result: AnalyzeResponse,
  tier: Exclude<NotificationTier, 'silent'>,
): void {
  // Replace any banner already on screen (e.g. after an SPA route change).
  removeBanner();

  const host = el('div', { attrs: { id: HOST_ID } });
  // Critical inline styles: correct placement + a hard style boundary before
  // the linked stylesheet loads (prevents flash / page-style bleed).
  host.style.cssText =
    'all: initial; position: fixed; top: 0; left: 0; right: 0; z-index: 2147483647;';

  const shadow = host.attachShadow({ mode: 'open' });

  const link = el('link', {
    attrs: { rel: 'stylesheet', href: chrome.runtime.getURL('content/content_script.css') },
  });
  shadow.appendChild(link);

  const banner = buildBanner(result, tier);
  shadow.appendChild(banner);

  // documentElement is always present, even before <body> is parsed.
  document.documentElement.appendChild(host);
  currentHost = host;

  // Escape closes the banner from anywhere on the page.
  escapeHandler = (e: KeyboardEvent) => {
    if (e.key === 'Escape') removeBanner();
  };
  document.addEventListener('keydown', escapeHandler, true);

  // For the high-severity alert, move focus to the banner so screen-reader and
  // keyboard users land on it. Caution stays polite and doesn't steal focus.
  if (tier === 'alert') {
    banner.focus({ preventScroll: true });
  }

  if (TIER_COPY[tier].autoDismiss) {
    startAutoDismiss(banner, CAUTION_AUTO_DISMISS_MS);
  }
}

/* ============================================================================
 * Message wiring (registered once per frame)
 * ========================================================================== */

function init(): void {
  chrome.runtime.onMessage.addListener((message: RuntimeMessage) => {
    if (message.type === MessageType.AnalysisResult) {
      const { result, tier } = message as AnalysisResultMessage;
      renderBanner(result, tier);
    }
    // This script never sends async responses; return nothing.
  });
}

if (!window.__adisContentScriptLoaded) {
  window.__adisContentScriptLoaded = true;
  init();
}

export {};
