# Plan to Resolve UI Authentication Issues

## Problem Diagnostic
1. The frontend attempts to authenticate against the backend `/api/auth/login` and `/api/auth/signup` endpoints.
2. The FastAPI backend fails to start up cleanly because of a `ModuleNotFoundError` for the `passlib` module when importing `backend.auth.service`. This causes the Vite proxy to return "Request failed" (Connection Refused) when the frontend makes API calls to these routes.
3. The virtual environment is missing `passlib` and `bcrypt`.
4. Additionally, the system does not have a local PostgreSQL server running, so even if the modules were importable, connection to PostgreSQL via the hardcoded `postgresql+psycopg2` driver would fail.

## Proposed Changes
We will proceed with the following steps to make authentication work:

### 1. Install Missing Python Dependencies
We will ask the user to run, or execute with the user's permission, the installation of:
- `passlib` (with bcrypt support)
- `bcrypt`
- `psycopg2-binary` (optional/required for PostgreSQL backend connection)

### 2. Implement SQLite Fallback in Backend Auth DB
If a local PostgreSQL database is not running or fails to connect, we will fallback to a local SQLite database (`auth.db` in the repository root or inside `.mitra/`).
We will update `backend/auth/db.py` to catch database connection/initialization errors (`OperationalError`) and automatically fall back to `sqlite:///auth.db` (or a configured path).

### 3. Add Settings to config.ini
We will add `[authdb]` section configurations to `config.ini` to avoid hardcoding defaults, including a fallback database URL or file path configuration if SQLite is used.

---

## Detailed Step-by-Step Implementation

### Step 1: Install Dependencies
Execute:
```bash
/home/sujithma/venv/bin/python -m pip install passlib bcrypt psycopg2-binary
```

### Step 2: Update `config.ini`
Add configuration settings under `[authdb]` in `config.ini`:
```ini
[authdb]
DB_HOST_ENV = AUTHDB_HOST
DB_PORT_ENV = AUTHDB_PORT
DB_NAME_ENV = AUTHDB_NAME
DB_USER_ENV = AUTHDB_USER
DB_PASSWORD_ENV = AUTHDB_PASSWORD
USER_WORKSPACE_ROOT = mitra
PASSWORD_MIN_LENGTH = 8
FALLBACK_DB_URL = sqlite:///auth.db
```

### Step 3: Update `backend/config_loader.py`
Add `fallback_db_url` attribute to `AuthDbConfig` dataclass and load it from `config.ini`.

### Step 4: Update `backend/auth/db.py`
Modify `engine()` method to catch `OperationalError` when initializing connection/tables and fallback to SQLite if PostgreSQL fails.
Add verbose logging to show which database backend (PostgreSQL vs SQLite) is being used.

### Step 5: Test the changes
1. Start backend server: `bin/mitra backend`
2. Run automated tests or test login/signup manually in the UI.
