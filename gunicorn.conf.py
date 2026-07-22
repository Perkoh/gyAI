# ============================================================
# ADIS — Gunicorn configuration
# Loaded via:  gunicorn -c gunicorn.conf.py "api.app:create_app()"
#
# Every value is overridable by an environment variable so you can
# tune the server on Fly.io (or in docker-compose) without a rebuild.
# ============================================================
import os

# --- Networking ---
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"

# --- Concurrency ---
# The analyze pipeline is I/O-bound during network feature extraction
# (DNS + WHOIS, up to ~3s), so gthread workers with several threads
# give better throughput per worker than plain sync workers.
#
# NOTE: the blueprint targets "4 workers". On a single shared vCPU with
# LightGBM + SHAP loaded per worker, 4 full workers is memory-heavy.
# Default here is 2 workers x 4 threads (tune via env). Raise workers
# only after confirming the VM has enough RAM (see fly.toml).
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")

# --- Timeouts ---
# timeout must comfortably exceed the worst-case WHOIS/DNS timeout (3s)
# plus model + SHAP time.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "10"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# --- Memory hygiene ---
# preload_app loads the app (and the LightGBM model singleton) ONCE in
# the master process, then forks workers. Forked workers share the
# model's memory pages via copy-on-write -> big RAM saving vs. each
# worker loading its own copy. This is what makes the model affordable
# on a small VM.
#
# preload also means api/extensions.py::_init_redis runs in the master
# and PINGS Redis there, so a live socket is inherited by every fork.
# TCP sockets aren't safe to share across processes, so the post_fork
# hook below drops the inherited pool in each worker (redis-py reopens
# lazily). See post_fork() at the bottom of this file.
preload_app = os.getenv("GUNICORN_PRELOAD", "true").lower() == "true"

# Recycle workers periodically to bound any slow memory creep from SHAP.
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "100"))

# --- Logging (stdout/stderr -> captured by Fly / Docker) ---
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()


# ============================================================
# Fork-safety hooks (only relevant when preload_app is True)
# ============================================================
def post_fork(server, worker):
    """
    Reset inherited network connections in each freshly-forked worker.

    With preload_app=True the Flask app is built once in the master process
    (so LightGBM + SHAP + the model load once and are shared copy-on-write).
    But api/extensions.py::_init_redis pings Redis during create_app(), which
    opens a real socket in the master. That socket is inherited by every
    worker, and sharing one connection across processes causes cross-talk.
    We drop the inherited pools here; redis-py transparently reopens a fresh
    connection per worker on next use. Hooks must never raise, so everything
    is wrapped defensively.
    """
    # 1) ADIS domain-cache Redis client (definitely opened in the master via
    #    the startup ping in _init_redis). This is the important one.
    try:
        from api.extensions import get_redis

        client = get_redis()
        if client is not None:
            client.connection_pool.disconnect()
            server.log.info("[post_fork] reset ADIS cache Redis connection pool")
    except Exception as exc:  # pragma: no cover
        server.log.warning(f"[post_fork] cache Redis reset skipped: {exc}")

    # 2) flask-limiter's Redis storage, if Redis-backed. Best-effort: the
    #    internal attribute path varies across flask-limiter / limits
    #    versions, and a miss is non-critical here because the limiter is
    #    configured with RATELIMIT_SWALLOW_ERRORS + in-memory fallback.
    try:
        from api.extensions import limiter

        storage = getattr(limiter, "storage", None)
        raw = getattr(storage, "storage", None)      # limits RedisStorage.storage
        pool = getattr(raw, "connection_pool", None)
        if pool is not None:
            pool.disconnect()
            server.log.info("[post_fork] reset flask-limiter Redis connection pool")
    except Exception as exc:  # pragma: no cover
        server.log.warning(f"[post_fork] limiter Redis reset skipped: {exc}")

    # NOTE on Supabase: the supabase-py client is httpx-based and opens its
    # sockets lazily on the first HTTP request (which always happens
    # post-fork, per request), so it doesn't need resetting here. If you ever
    # make it eager (e.g. a health-check call inside create_app), add its
    # reset to this hook the same way.

def worker_abort(worker):
    """
    Fires when a worker is killed for exceeding `timeout` — i.e. a request
    ran longer than the worker-timeout backstop (default 10s via GUNICORN_TIMEOUT).

    This is your early-warning signal that a request hung, almost always on a
    slow DNS/WHOIS lookup during network feature extraction. A steady trickle
    of these means the per-lookup network timeout in features/network.py should
    be tightened — the Gunicorn timeout is only the last-resort net, not the
    real control. Hooks must never raise, so this only logs.
    """
    worker.log.warning(
        "[worker_abort] worker pid=%s aborted — a request exceeded the "
        "gunicorn timeout (%ss). Likely a hung DNS/WHOIS lookup; check "
        "features/network.py per-lookup timeouts if these recur.",
        worker.pid,
        timeout,
    )