# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Plotly Dash web app (Johnson Lab, Yale) for exploring gut microbiome / metabolomics data. Backed by PostgreSQL with a layered cache (in-memory + Redis Cloud). Deployed via gunicorn (Heroku-style — see `Procfile`).

The app serves a single user-facing route: `/yale-university`. `/` 301-redirects to it; anything else renders `layout404.py`.

## Commands

```bash
# Install deps (Python 3.x, expects a venv)
pip install -r requirements.txt

# Run locally (loads .env, starts dev server with debug=True, no reloader)
python app.py

# Production-style run
gunicorn app:server --timeout 60 --workers 2 --worker-class gthread --threads 4

# Smoke-test a couple of expensive query paths (NOT a real test suite — needs a live DB)
python test_performance.py

# Drop a Postgres table (interactive prompt; uses DATABASE_URL)
python delete_table.py
```

There is no linter, formatter, or unit test framework configured. `test_performance.py` is a manual timing script, not pytest.

## Required environment

- `DATABASE_URL` — PostgreSQL connection string. Without it, all data functions error out.
- `REDISCLOUD_URL` — Redis Cloud URL. Falls back to `redis://localhost:6379/0` if unset; if Redis is unreachable the cache silently degrades to the in-process memory cache.

`.env` is loaded via `python-dotenv` from the repo root.

## Architecture

The codebase is small but the call graph through caching layers is non-obvious. The flow is:

```
app.py (entry, Dash app + startup warm-up)
  └── layout.py (massive single-file UI — tabs, dropdowns, dcc.Graph placeholders)
  └── compare_tumor/callback.py (register_callbacks — wires every Output to a query+plot)
        └── compare_tumor/data_functions.py (DB queries, connection pool, cache decorators)
        └── compare_tumor/dynamicPlots.py (plotly figure builders + PlotConfig styling)
        └── compare_tumor/simple_redis_cache.py (Redis cache + memory fallback)
```

### Caching is layered — order matters

Most heavy data functions in `data_functions.py` stack **two** cache decorators, e.g.:

```python
@simple_cache(max_size=50, ttl=600)      # in-process, per-worker, TTL-based
@simple_redis_cache()                     # cross-worker, Redis with native eviction
def get_gmm_name(table_name): ...
```

Decorator order is significant: `simple_cache` is applied last so it wraps the Redis-cached call. A hit in the in-process cache short-circuits before Redis is consulted. When debugging stale results, clear **both** layers — `func.clear_cache()` only clears the outermost decorator.

`compare_tumor/cache_utils.py` contains a third, disk-based `page_cache` (joblib) that is defined but not wired into the main data path — leave it alone unless explicitly working on it.

### Database access

All queries go through `db_pool` (a singleton `psycopg2.pool.ThreadedConnectionPool`, 3–15 conns) via the `get_db_connection()` context manager in `data_functions.py`. Do **not** call `psycopg2.connect()` directly in new code — `delete_table.py` is the one intentional exception (admin script, runs outside the app).

The pool is initialized in `app.py:initialize_performance_optimizations()` at import time, which also pre-warms caches for the two known tables: `gmm_test_1` and `in_vivo`. If you add a new table that should be hot at startup, add it to `warm_cache()` in `app.py`.

### Callbacks

`compare_tumor/callback.py` is a single ~55KB file that registers every Dash callback via `register_callbacks(app)`. Each callback is decorated with `@performance_logger` and writes to `app.log`. The `region` list (`cecum`, `ascending`, `transverse`, `descending`, `sigmoid`, `rectosigmoid`, `rectum`) is duplicated across `app.py`, `layout.py`, `callback.py`, and `dynamicPlots.py` — keep them in sync if changing.

### Plot styling

`PlotConfig` in `dynamicPlots.py` is the central styling source (colors, layout defaults, box/scatter settings). Region colors live separately in `compare_tumor/constant.py:region_colors`. Prefer `create_dynamic_scatter_plot` / `tumor_vs_normal_plot` / `all_regions_plots` over building figures inline in callbacks.

## Conventions worth knowing

- Dev tools and hot reload are explicitly **disabled** in `app.py` (`enable_dev_tools(debug=False, dev_tools_hot_reload=False)`) to avoid WebSocket errors in production. `app.run_server(debug=True, use_reloader=False)` in `__main__` is for local only.
- `app.log` is gitignored but written to by the logging config in both `data_functions.py` and `callback.py` — tail it when diagnosing performance issues.
- The redundant Redis-client setup that used to live in `layout.py` has been removed (see commit `d418d25`); the only Redis client lives in `app.py`. Don't reintroduce a second one.
- The three large markdown docs at the repo root (`IMPLEMENTATION_SUMMARY.md`, `PERFORMANCE_OPTIMIZATIONS.md`, `REDIS_*_GUIDE.md`) are historical write-ups of past optimization work, not active design docs. Use them for context, not as a spec.
