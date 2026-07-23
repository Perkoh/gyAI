/**
 * gyAI — Toolbar Popup Logic (extension/src/popup/popup.ts)
 * ========================================================
 *
 * Drives the browser-action popup (blueprint §11.3). Shows the active tab's
 * current verdict, lets the user report the site, and exposes settings
 * (protection on/off, caution banners, API endpoint, clear cache).
 *
 * All page/site analysis is owned by the service worker; the popup only:
 *   • asks the worker for the active tab's status (GetCurrentStatus → §11.3),
 *   • relays feedback (SubmitFeedback → §10.1),
 *   • asks the worker to clear the local cache (ClearCache → §12.2),
 *   • reads/writes {@link ExtensionSettings} directly in chrome.storage.local.
 *
 * Settings note: there is no settings message in the runtime protocol, and the
 * worker re-reads settings from storage on every navigation, so the popup writes
 * them straight to storage under {@link SETTINGS_KEY}. That key MUST stay in
 * sync with the service worker's `SETTINGS_KEY` ('adis:settings').
 *
 * Depends only on ../utils/types.
 */

import {
  ClearCacheMessage,
  CurrentStatusMessage,
  DEFAULT_SETTINGS,
  ExtensionSettings,
  FeedbackAck,
  FeedbackVerdict,
  GetCurrentStatusMessage,
  MessageType,
  RiskLabel,
  RuntimeMessage,
  SubmitFeedbackMessage,
} from '../utils/types';

/** chrome.storage key for settings — must match the service worker's constant. */
const SETTINGS_KEY = 'adis:settings';

/* ============================================================================
 * Thin promise wrappers (callback form → Chrome MV3 + Firefox)
 * ========================================================================== */

function sendMessage<T = unknown>(message: RuntimeMessage): Promise<T | undefined> {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(message, (response) => {
        void chrome.runtime?.lastError;
        resolve(response as T);
      });
    } catch {
      resolve(undefined);
    }
  });
}

function storageGet(keys: string | string[] | null): Promise<Record<string, unknown>> {
  const area = chrome.storage?.local;
  if (!area) return Promise.resolve({});
  return new Promise((resolve) => {
    area.get(keys, (items) => {
      void chrome.runtime?.lastError;
      resolve((items ?? {}) as Record<string, unknown>);
    });
  });
}

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

/* ============================================================================
 * Settings (read/merge/write)
 * ========================================================================== */

async function loadSettings(): Promise<ExtensionSettings> {
  const items = await storageGet(SETTINGS_KEY);
  const stored = items[SETTINGS_KEY] as Partial<ExtensionSettings> | undefined;
  return { ...DEFAULT_SETTINGS, ...(stored ?? {}) };
}

async function patchSettings(patch: Partial<ExtensionSettings>): Promise<ExtensionSettings> {
  const merged = { ...(await loadSettings()), ...patch };
  await storageSet({ [SETTINGS_KEY]: merged });
  return merged;
}

/* ============================================================================
 * DOM helpers
 * ========================================================================== */

function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`[gyAI popup] missing element #${id}`);
  return node as T;
}

/** Presentation per risk label (§11.3). */
const STATUS_UI: Record<RiskLabel, { icon: string; text: string; tone: string }> = {
  safe: { icon: '✅', text: 'Safe', tone: 'safe' },
  suspicious: { icon: '⚠', text: 'Suspicious', tone: 'caution' },
  malicious: { icon: '🚨', text: 'High Risk', tone: 'alert' },
};

/* ============================================================================
 * Status rendering
 * ========================================================================== */

/** Holds the active tab's resolved status for use by the report action. */
let activeDomain: string | null = null;
let activeLabel: RiskLabel = 'safe';

function renderCard(html: DocumentFragment | HTMLElement, tone: string): void {
  const card = $('status-card');
  card.className = `adis-card adis-card--${tone}`;
  card.setAttribute('aria-busy', 'false');
  card.replaceChildren(html);
}

/** Build the normal "site + status + score" card body. */
function statusBody(domain: string, icon: string, statusText: string, pct: number): HTMLElement {
  const wrap = document.createElement('div');

  const siteLabel = document.createElement('p');
  siteLabel.className = 'adis-card__label';
  siteLabel.textContent = 'Current site';

  const siteName = document.createElement('p');
  siteName.className = 'adis-card__domain';
  siteName.textContent = domain; // textContent: never parse as HTML

  const statusRow = document.createElement('p');
  statusRow.className = 'adis-card__status';
  const iconEl = document.createElement('span');
  iconEl.className = 'adis-card__status-icon';
  iconEl.setAttribute('aria-hidden', 'true');
  iconEl.textContent = icon;
  const statusStr = document.createElement('span');
  statusStr.textContent = statusText;
  statusRow.append(iconEl, statusStr);

  const scoreRow = document.createElement('p');
  scoreRow.className = 'adis-card__score';
  scoreRow.textContent = `Risk score: ${pct}%`;

  wrap.append(siteLabel, siteName, statusRow, scoreRow);
  return wrap;
}

/** Build a simple message-only card (unsupported / not-checked / off). */
function noticeBody(title: string, detail: string): HTMLElement {
  const wrap = document.createElement('div');
  const t = document.createElement('p');
  t.className = 'adis-card__notice-title';
  t.textContent = title;
  const d = document.createElement('p');
  d.className = 'adis-card__notice-detail';
  d.textContent = detail;
  wrap.append(t, d);
  return wrap;
}

async function refreshStatus(): Promise<void> {
  const settings = await loadSettings();

  // Master switch off (§1.1): make that explicit rather than showing a verdict.
  if (!settings.enabled) {
    activeDomain = null;
    ($('report-toggle') as HTMLButtonElement).hidden = true;
    renderCard(
      noticeBody('Protection is off', 'Turn it back on in Settings to analyze sites.'),
      'off',
    );
    return;
  }

  const reply = await sendMessage<CurrentStatusMessage>({
    type: MessageType.GetCurrentStatus,
  } satisfies GetCurrentStatusMessage);

  const domain = reply?.domain ?? null;
  const result = reply?.result ?? null;
  activeDomain = domain;

  const reportBtn = $('report-toggle') as HTMLButtonElement;

  if (!domain) {
    reportBtn.hidden = true;
    renderCard(
      noticeBody('Not available here', "gyAI doesn't run on browser or internal pages."),
      'off',
    );
    return;
  }

  if (!result) {
    // Domain is real but we have no cached verdict (e.g. page pre-dates install).
    reportBtn.hidden = false;
    activeLabel = 'safe';
    renderCard(
      noticeBody(domain, 'No analysis yet — reload the page to check it.'),
      'unknown',
    );
    return;
  }

  activeLabel = result.label;
  const ui = STATUS_UI[result.label];
  const pct = Math.round(result.score * 100);
  reportBtn.hidden = false;
  renderCard(statusBody(domain, ui.icon, ui.text, pct), ui.tone);
}

/* ============================================================================
 * Report flow
 * ========================================================================== */

function wireReport(): void {
  const toggle = $('report-toggle') as HTMLButtonElement;
  const panel = $('report-panel');
  const statusLine = $('report-status');

  toggle.addEventListener('click', () => {
    const show = panel.hasAttribute('hidden');
    panel.toggleAttribute('hidden', !show);
    toggle.setAttribute('aria-expanded', String(show));
  });

  for (const btn of Array.from(panel.querySelectorAll<HTMLButtonElement>('.adis-choice'))) {
    btn.addEventListener('click', async () => {
      if (!activeDomain) return;
      const verdict = btn.dataset.verdict as FeedbackVerdict;

      panel.querySelectorAll<HTMLButtonElement>('.adis-choice').forEach((b) => (b.disabled = true));
      statusLine.textContent = 'Sending…';

      const ack = await sendMessage<FeedbackAck>({
        type: MessageType.SubmitFeedback,
        feedback: {
          domain: activeDomain,
          system_label: activeLabel,
          user_verdict: verdict,
        },
      } satisfies SubmitFeedbackMessage);

      statusLine.textContent = ack?.success
        ? 'Thanks — your report was sent.'
        : "That didn't go through. Please try again.";

      if (!ack?.success) {
        panel.querySelectorAll<HTMLButtonElement>('.adis-choice').forEach((b) => (b.disabled = false));
      }
    });
  }
}

/* ============================================================================
 * Settings flow
 * ========================================================================== */

async function wireSettings(): Promise<void> {
  const settings = await loadSettings();

  const panel = $('settings-panel');
  const toggle = $('settings-toggle') as HTMLButtonElement;
  const enabled = $('set-enabled') as HTMLInputElement;
  const caution = $('set-caution') as HTMLInputElement;
  const apiUrl = $('set-apiurl') as HTMLInputElement;
  const apiErr = $('apiurl-error');
  const clearBtn = $('clear-cache') as HTMLButtonElement;
  const clearStatus = $('clear-status');

  // Seed current values.
  enabled.checked = settings.enabled;
  caution.checked = settings.showCautionBanners;
  apiUrl.value = settings.apiBaseUrl;

  toggle.addEventListener('click', () => {
    const show = panel.hasAttribute('hidden');
    panel.toggleAttribute('hidden', !show);
    toggle.setAttribute('aria-expanded', String(show));
  });

  // Toggling protection re-renders the status card immediately.
  enabled.addEventListener('change', async () => {
    await patchSettings({ enabled: enabled.checked });
    await refreshStatus();
  });

  caution.addEventListener('change', async () => {
    await patchSettings({ showCautionBanners: caution.checked });
  });

  // API endpoint: must be HTTPS (§FLAG 5). Save on change if valid.
  apiUrl.addEventListener('change', async () => {
    const value = apiUrl.value.trim();
    const valid = /^https:\/\/.+/i.test(value);
    apiErr.hidden = valid;
    if (valid) await patchSettings({ apiBaseUrl: value.replace(/\/+$/, '') });
  });

  clearBtn.addEventListener('click', async () => {
    clearBtn.disabled = true;
    clearStatus.textContent = 'Clearing…';
    const ack = await sendMessage<{ success: boolean }>({
      type: MessageType.ClearCache,
    } satisfies ClearCacheMessage);
    clearStatus.textContent = ack?.success ? 'Local cache cleared.' : 'Could not clear cache.';
    clearBtn.disabled = false;
  });
}

/* ============================================================================
 * Docs link
 * ========================================================================== */

function wireDocsLink(): void {
  $('docs-link').addEventListener('click', async () => {
    const settings = await loadSettings();
    // Derive the docs origin from the configured API base (strip /api/v1).
    let href = settings.apiBaseUrl;
    try {
      href = new URL(settings.apiBaseUrl).origin;
    } catch {
      /* fall back to the raw value */
    }
    chrome.tabs.create({ url: href });
  });
}

/* ============================================================================
 * Boot
 * ========================================================================== */

async function main(): Promise<void> {
  wireReport();
  wireDocsLink();
  await wireSettings();
  await refreshStatus();
}

document.addEventListener('DOMContentLoaded', () => void main());
