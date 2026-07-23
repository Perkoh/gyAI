# gyAI Privacy Policy

**Last updated:** 23/07/2026

gyAI is a browser extension that warns you when you visit a domain that may be
suspicious or malicious. This policy explains exactly what data the extension
handles, what leaves your browser, and what happens to it.

## Summary

gyAI sends one thing to our servers: the **bare domain name** of sites you
visit. It never sends the full web address, never reads page content, and never
sends cookies or any identifier that could link a domain to you personally.

## What we send

When you navigate to a website, the extension extracts only the hostname from
the address and sends it to our analysis service.

For example, if you visit:

```
https://example.com/account/settings?user=12345
```

the only thing transmitted is:

```
example.com
```

The path (`/account/settings`), the query string (`?user=12345`), and the
fragment are discarded inside your browser and never leave it.

If you choose to submit a report using the "Report This Site" button, we also
receive the domain you reported, the risk label gyAI assigned it, your verdict
(safe / dangerous / unsure), and any optional comment you type.

## What we do not collect

- Full URLs, paths, query strings, or fragments
- Page content, text, form data, or anything you type on a website
- Cookies — API requests are sent with cookies explicitly omitted
- Names, email addresses, account identifiers, or login credentials
- IP-based profiles, advertising identifiers, or tracking cookies
- Browsing history as a linked sequence — each domain is analyzed
  independently, and we do not build a profile of your browsing

gyAI has no user accounts and requires no sign-up.

## Data stored on your device

The extension stores the following locally using `chrome.storage.local`. This
data stays on your computer and is never uploaded:

- **Analysis results**, cached for up to one hour so that revisiting the same
  domain does not trigger a repeat network request
- **Your settings** — protection on/off, whether caution banners are shown, and
  the API endpoint

You can erase all cached results at any time using **Clear cached results** in
the extension's Settings panel. Uninstalling the extension removes everything
stored locally.

## Data stored on our servers

Domains that our model scores as suspicious or malicious are logged so we can
monitor accuracy and improve the model. Domains scored as safe are not logged.

Log records contain the domain, the risk score, the model version, and a
timestamp. They do not contain any information identifying who requested the
analysis.

Reports you submit through the "Report This Site" button are stored so we can
review and correct model errors.

Retention: analysis records are kept for six months. Feedback reports
are kept for six months.

## How the data is used

Domain data is used solely to:

1. Return a risk assessment for the domain you are visiting
2. Monitor the accuracy of the model and improve it over time

## What we never do

- We do not sell your data to anyone
- We do not share it with third parties for advertising or marketing
- We do not use it for creditworthiness, lending, or any financial assessment
- We do not use it for any purpose unrelated to warning you about risky domains

## Service providers

Our analysis service runs on Fly.io, and our data is stored using Supabase.
These providers process data on our behalf under their own security and privacy
commitments.

## Your choices

- **Turn protection off** at any time from the extension's Settings panel. When
  off, no domains are analyzed and nothing is transmitted.
- **Clear your local cache** from the Settings panel.
- **Uninstall the extension** to stop all data collection immediately and remove
  all locally stored data.

## Children

gyAI is not directed at children under 13 and we do not knowingly collect data
from them.

## Changes to this policy

If we change how the extension handles data, we will update this page and revise
the "Last updated" date above.

## Contact

Questions about this policy or about your data: perkohawuah@gmail.com
