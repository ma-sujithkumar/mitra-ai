# SPEC: Authentication Gate (Frontend + Backend)

Source requirements: [frontend/requirements.txt](requirements.txt)

This spec describes adding an authentication page that gates the existing
dashboard / UI, backed by a PostgreSQL auth database. It reuses existing
patterns: the FastAPI router/`Depends` style in [backend/routers](../backend/routers),
the `requestJson` API helper in [frontend/src/api/client.js](src/api/client.js),
and the screen/route switch in [frontend/src/App.jsx](src/App.jsx).

---

## 1. Goals

1. Show an **Auth page before** the dashboard / existing UI loads.
2. Support **Login** (default) and **Signup** modes with a mode toggle.
3. On successful auth, route into the existing UI (`Dashboard`).
4. Persist users in **PostgreSQL** (`authdb`) inside the code directory.
5. Create a per-user workspace folder `mitra/<user_id>` on first login.

### Non-goals
- Forgot-password recovery flow (explicitly out of scope; show contact popup).
- Changing any existing backend behaviour. Auth is added as an **isolated new
  module/router**; existing routers are not modified.

---

## 2. Frontend

All auth code lives in a dedicated folder: `frontend/src/auth/`.

### 2.1 Files
```
frontend/src/auth/
  AuthPage.jsx        # Container: mode state, layout, popup
  LoginForm.jsx       # Login mode fields + submit
  SignupForm.jsx      # Signup mode fields + validation
  ForgotPasswordModal.jsx
  authApi.js          # login()/signup() wrappers over requestJson
  authValidation.js   # email + password validators (pure functions)
  auth.css            # Scoped styles (reuse theme.css tokens)
```

### 2.2 Behaviour

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Two text fields: username, password | Reuse [FormField](src/components/FormField.jsx); password input `type="password"` |
| 2 | Two buttons: Login, Signup | Mode toggle via [Segmented](src/components/Segmented.jsx) |
| 3 | Default = login mode | `useState('login')` |
| 4 | Signup inputs: name, username, password + validation | `SignupForm` with `authValidation.js` checks |
| 5 | Login inputs: username, password | `LoginForm` (note: requirements line 5 says "name and password" but contradicts line 1; login uses **username + password** as the canonical key) |
| 6 | On success -> dashboard | Call `App.onAuthenticated(user)`; render existing screens |
| 7 | Duplicate username -> "user id already exists" | Surface backend `409` error message |
| 8 | Create `mitra/<user_id>` | Backend creates folder on login/signup success |
| 9 | Forgot password popup | `ForgotPasswordModal` with fixed contact text |

### 2.3 Validation (`authValidation.js`)
- **Username/email**: if username is an email, validate with
  `^[^\s@]+@[^\s@]+\.[^\s@]+$`. Otherwise require length >= 3, alnum + `._-`.
- **Password**: min 8 chars, at least one letter and one digit. Return a list of
  human-readable failures; block submit until empty.
- Pure functions returning `{ valid: boolean, errors: string[] }` (unit-testable
  with `node --test`, matching [src/trainingState.test.js](src/trainingState.test.js)).

### 2.4 Forgot-password popup text
> Forgot-password feature is not implemented. If you forgot your id and
> password, contact deeplearning1227@gmail.com or create a new id.

### 2.5 App integration ([src/App.jsx](src/App.jsx))
- Add `authUser` state, initialised from `localStorage('mitra.authUser')`.
- If `!authUser`, render `<AuthPage onAuthenticated={...} />` instead of the
  `Sidebar`/`workspace` layout.
- On success, persist user and render existing screens unchanged.
- Add a logout control (clears state + storage) in [TopBar](src/components/TopBar.jsx)
  or [Sidebar](src/components/Sidebar.jsx).

### 2.6 API wrappers (`authApi.js`)
Built on `requestJson` from [src/api/client.js](src/api/client.js):
- `signup({ name, username, password })` -> `POST /api/auth/signup`
- `login({ username, password })` -> `POST /api/auth/login`

---

## 3. Backend

New isolated module under `backend/` (no edits to existing routers except one
`include_router` line in [backend/main.py](../backend/main.py)).

### 3.1 Files
```
backend/auth/
  __init__.py
  models.py        # SQLAlchemy User model (id, name, username, password_hash, created_at)
  db.py            # Engine/session factory from config.ini [authdb]
  service.py       # AuthService: signup/login, hashing, workspace creation
  schemas.py       # Pydantic request/response models
backend/routers/
  auth.py          # APIRouter(prefix="/api/auth")
```

### 3.2 Endpoints (`backend/routers/auth.py`)
| Method | Path | Body | Success | Error |
|--------|------|------|---------|-------|
| POST | `/api/auth/signup` | `{name, username, password}` | `201 {user_id, username, name}` | `409 {detail:{message:"user id already exists"}}` |
| POST | `/api/auth/login` | `{username, password}` | `200 {user_id, username, name}` | `401 {detail:{message:"invalid credentials"}}` |

Error shape matches the frontend's `parseJsonResponse` contract
(`payload.detail.message`) in [src/api/client.js](src/api/client.js).

### 3.3 Auth logic (`service.py`)
- Hash passwords with `passlib[bcrypt]` (no plaintext storage).
- Signup: reject duplicate `username` -> `409`.
- On signup/login success: `os.makedirs(mitra/<user_id>, exist_ok=True)`
  (the `mkdir -p` rule), under the path from config (`[authdb] USER_WORKSPACE_ROOT`).
- `user_id` = stable identifier (DB primary key or the username, decided in impl).

### 3.4 PostgreSQL (`authdb`)
- Connection driven entirely from `config.ini` (no hardcoded paths/creds).
- `db.py` builds the SQLAlchemy URL from the `[authdb]` section.
- DB credentials/host come from `.env` (reusing the existing dotenv pattern in
  [backend/routers/health.py](../backend/routers/health.py)), referenced by name
  in config; **no secrets committed**.
- Tables created on startup via `Base.metadata.create_all` (or Alembic if added).

### 3.5 config.ini additions ([config.ini](../config.ini))
```ini
[authdb]
DB_HOST_ENV=AUTHDB_HOST
DB_PORT_ENV=AUTHDB_PORT
DB_NAME_ENV=AUTHDB_NAME
DB_USER_ENV=AUTHDB_USER
DB_PASSWORD_ENV=AUTHDB_PASSWORD
USER_WORKSPACE_ROOT=mitra
PASSWORD_MIN_LENGTH=8
```
Corresponding `.env` keys (not committed): `AUTHDB_HOST`, `AUTHDB_PORT`,
`AUTHDB_NAME`, `AUTHDB_USER`, `AUTHDB_PASSWORD`.

### 3.6 Dependencies ([requirements.txt](../requirements.txt))
Add: `sqlalchemy>=2.0`, `psycopg2-binary>=2.9`, `passlib[bcrypt]>=1.7`.

### 3.7 main.py wiring ([backend/main.py](../backend/main.py))
Single added line: `app.include_router(auth.router)`. CORS already allows the
Vite dev origins.

---

## 4. Security
- Passwords hashed (bcrypt); never stored or logged in plaintext.
- Parameterised queries via SQLAlchemy ORM (no SQL injection).
- Generic `401` on login failures (no username enumeration on login).
- No secrets in `config.ini` or source; only env-var **names** are referenced.
- Validate/normalise `user_id` before building `mitra/<user_id>` to prevent path
  traversal.

---

## 5. Testing
- **Frontend**: `authValidation.test.js` via `node --test` (add to `test` script
  in [package.json](package.json)), covering email/password edge cases.
- **Backend**: pytest in [backend/tests](../backend/tests):
  - signup creates user + `mitra/<user_id>` folder,
  - duplicate username -> 409 "user id already exists",
  - login success/failure,
  - password is hashed (not plaintext).
- Use a test/sqlite or transactional fixture so tests don't require a live
  Postgres instance in CI.

---

## 6. Acceptance criteria
1. Visiting the app shows the Auth page (login default) before any dashboard UI.
2. Signup validates email + password and rejects duplicates with the exact
   message "user id already exists".
3. Successful login/signup routes to the existing dashboard.
4. `mitra/<user_id>` exists on disk after first successful auth.
5. Forgot-password popup shows the required contact message.
6. Passwords are stored hashed in PostgreSQL `authdb`.
7. No existing backend route behaviour changes.

---

## 7. Open questions
1. Requirements line 5 ("login mode - name and password") conflicts with line 1
   (username field). This spec uses **username + password** for login. Confirm.
2. Should `user_id` be the DB integer id or the username for the `mitra/<user_id>`
   folder name? (Spec assumes username, sanitised.)
3. Session/token strategy after login (cookie/JWT vs. client-side flag only).
   Current scope uses a client-side `localStorage` flag; confirm if a real
   session token is required.
