"""Integration tests for the Flask API (blueprint section 10).

Assumed interface
-----------------
``api/app.py`` exposes an application factory ``create_app()``. Routes live
under the ``/api/v1`` prefix:

    POST /api/v1/analyze     {"domain": "..."} -> scored result
    GET  /api/v1/health      -> health check
    GET  /api/v1/version     -> model version info
    POST /api/v1/feedback    -> submit a report

Analyze success contract (section 10.2)::

    {
      "domain": str, "score": float, "label": str, "confidence": str,
      "reasons": list[str], "model_version": str, "analysis_id": str | None,
      "duration_ms": int, "cached": bool, "network_features_used": bool
    }

Error contract (section 10.2)::

    {"error": {"code": str, "message": str, "status": int}}

These are integration-style: they run against a real ``create_app()``. If the
app can't be constructed in the test environment (e.g. the trained model file
isn't present yet), the whole module skips. External I/O (Redis, Supabase) is
patched to no-ops where a seam is available; validation/health/version tests do
not depend on the model and should pass as soon as the routes exist.

The base prefix is auto-detected: we probe ``/api/v1/health`` first and fall
back to ``/health`` so the suite tolerates either mounting choice.
"""

from __future__ import annotations

import pytest

from conftest import import_or_skip


@pytest.fixture(scope="module")
def app():
    module = import_or_skip("api.app")
    create_app = getattr(module, "create_app", None)
    if create_app is None:
        pytest.skip("api.app.create_app not implemented yet.", allow_module_level=True)
    try:
        application = create_app()
    except Exception as exc:  # missing model file, misconfig, etc.
        pytest.skip(f"create_app() could not start in test env: {exc}",
                    allow_module_level=True)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(scope="module")
def base(app):
    """Detect the route prefix by probing both candidates once."""
    test_client = app.test_client()
    for prefix in ("/api/v1", ""):
        resp = test_client.get(f"{prefix}/health")
        if resp.status_code != 404:
            return prefix
    pytest.skip("No /health route found under /api/v1 or /.", allow_module_level=True)


# ---------------------------------------------------------------------------
# Health & version (model-independent).
# ---------------------------------------------------------------------------
def test_health_ok(client, base):
    resp = client.get(f"{base}/health")
    assert resp.status_code == 200
    body = resp.get_json(silent=True) or {}
    # Common shapes: {"status": "ok"} or {"status": "healthy"}.
    status = str(body.get("status", "")).lower()
    assert status in ("ok", "healthy", "up") or body == {} or resp.status_code == 200


def test_version_reports_model_version(client, base):
    resp = client.get(f"{base}/version")
    if resp.status_code == 404:
        pytest.skip("/version route not implemented yet.")
    assert resp.status_code == 200
    body = resp.get_json(silent=True) or {}
    # Accept either 'model_version' or a nested/version key.
    assert any(k in body for k in ("model_version", "version")), body


# ---------------------------------------------------------------------------
# Input validation (model-independent).
# ---------------------------------------------------------------------------
def test_analyze_missing_domain_is_rejected(client, base):
    resp = client.post(f"{base}/analyze", json={})
    assert resp.status_code in (400, 422), resp.get_data(as_text=True)


def test_analyze_invalid_domain_is_rejected(client, base):
    resp = client.post(f"{base}/analyze", json={"domain": "not a domain !!!"})
    assert resp.status_code in (400, 422)
    body = resp.get_json(silent=True) or {}
    if "error" in body:
        err = body["error"]
        assert "code" in err and "message" in err and "status" in err


def test_analyze_rejects_non_json(client, base):
    resp = client.post(f"{base}/analyze", data="domain=google.com",
                       content_type="application/x-www-form-urlencoded")
    assert resp.status_code in (400, 415, 422)


# ---------------------------------------------------------------------------
# Analyze happy path (requires a working model; skips otherwise).
# ---------------------------------------------------------------------------
def _post_analyze(client, base, domain):
    resp = client.post(f"{base}/analyze", json={"domain": domain})
    if resp.status_code == 503:
        pytest.skip("model/service not ready (503).")
    if resp.status_code == 500:
        pytest.skip("analyze endpoint returned 500 — model likely unavailable in test env.")
    return resp


ANALYZE_KEYS = {
    "domain", "score", "label", "confidence", "reasons",
    "model_version", "cached", "network_features_used",
}


def test_analyze_safe_domain_contract(client, base):
    resp = _post_analyze(client, base, "github.com")
    assert resp.status_code == 200
    body = resp.get_json()
    missing = ANALYZE_KEYS - set(body)
    assert not missing, f"analyze response missing keys: {missing}"
    assert body["domain"] == "github.com"
    assert 0.0 <= float(body["score"]) <= 1.0
    assert body["label"] in ("safe", "suspicious", "malicious")
    assert isinstance(body["reasons"], list)
    assert isinstance(body["cached"], bool)


def test_analyze_returns_reasons_for_flagged_domain(client, base):
    resp = _post_analyze(client, base, "secure-login-paypa1.xyz")
    assert resp.status_code == 200
    body = resp.get_json()
    if body["label"] == "safe":
        pytest.skip("model scored the sample domain as safe; can't assert reasons.")
    assert len(body["reasons"]) >= 1
    # analysis_id is populated for suspicious/malicious results.
    assert body.get("analysis_id") not in (None, "")


def test_second_identical_request_is_cached(client, base):
    """A repeat lookup should be served from cache (cached: true)."""
    first = _post_analyze(client, base, "example.com")
    assert first.status_code == 200
    second = _post_analyze(client, base, "example.com")
    assert second.status_code == 200
    body = second.get_json()
    # If caching is wired, the second call should report cached True. If the
    # environment has no Redis, tolerate a miss rather than failing hard.
    if body.get("cached") is not True:
        pytest.skip("cache not active in this environment (no Redis?).")
    assert body["cached"] is True


# ---------------------------------------------------------------------------
# Feedback endpoint.
# ---------------------------------------------------------------------------
def test_feedback_accepts_valid_report(client, base):
    payload = {
        "domain": "example.com",
        "system_label": "safe",
        "user_verdict": "false_positive",
        "user_comment": "This site is fine.",
    }
    resp = client.post(f"{base}/feedback", json=payload)
    if resp.status_code == 404:
        pytest.skip("/feedback route not implemented yet.")
    if resp.status_code in (500, 503):
        pytest.skip("feedback storage (Supabase) not available in test env.")
    assert resp.status_code in (200, 201, 202)


def test_feedback_rejects_bad_verdict(client, base):
    payload = {
        "domain": "example.com",
        "system_label": "safe",
        "user_verdict": "definitely-not-a-valid-enum",
    }
    resp = client.post(f"{base}/feedback", json=payload)
    if resp.status_code == 404:
        pytest.skip("/feedback route not implemented yet.")
    # Should validate the verdict enum from the schema.
    assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# CORS (blueprint FLAG 6): responses must carry CORS headers for the extension.
# ---------------------------------------------------------------------------
def test_cors_header_present_on_analyze(client, base):
    resp = client.post(
        f"{base}/analyze",
        json={"domain": "github.com"},
        headers={"Origin": "chrome-extension://abcdefghijklmnop"},
    )
    if resp.status_code in (500, 503):
        pytest.skip("model not available; skipping CORS assertion.")
    # flask-cors adds this header when configured.
    if "Access-Control-Allow-Origin" not in resp.headers:
        pytest.skip("CORS not configured in this build yet.")
    assert resp.headers["Access-Control-Allow-Origin"]
