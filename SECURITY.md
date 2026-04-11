# Security Policy

## Supported Versions

This is an active research project. Only the current `main` branch receives security updates. There are no versioned releases.

| Branch / Version | Supported |
| ---------------- | --------- |
| `main`           | ✅        |
| older commits    | ❌        |

## Scope

The following components are in scope for security reports:

- **FastAPI web service** (`web/`, `api/`) — REST endpoints for training control, inference, and data access
- **Database access** (`db/`) — PostgreSQL schema and queries; SQL injection risks
- **Docker Compose setup** — credential handling, exposed ports, container isolation
- **Python dependencies** (`trainer/`) — torch, mlflow, requests, and other packages
- **Frontend dependencies** (`web/`) — vite, react, and npm packages
- **Sensitive data exposure** — training logs, model weights, episode data stored in PostgreSQL

The following are **out of scope**:

- SNN model outputs or behaviour (ML model decisions are not security issues)
- Vulnerabilities in Docker, PostgreSQL, or MLflow themselves (report upstream)
- Issues requiring physical access to the host machine
- Performance degradation or training instability

## Reporting a Vulnerability

**Do not report security vulnerabilities via public GitHub Issues.**

Use GitHub's private vulnerability reporting instead:

👉 **[Report a vulnerability privately](https://github.com/nfriacowboy/bio-organism/security/advisories/new)**

Please include:
- A clear description of the vulnerability and its potential impact
- Steps to reproduce (minimal reproduction preferred)
- Affected component (`api/`, `db/`, `web/`, `trainer/`, `docker-compose.yml`, etc.)
- Any suggested fix if you have one

### What to expect

| Timeline | Action |
| -------- | ------ |
| 48–72 h  | Acknowledgement of the report |
| 7 days   | Initial assessment and severity classification |
| 30 days  | Fix or mitigation (for confirmed vulnerabilities) |
| After fix | Public disclosure coordinated with reporter |

Vulnerabilities confirmed as valid will receive a CVE via GitHub Security Advisories. Reports that are out of scope or not reproducible will be closed with an explanation.

## Security Configuration Notes

When self-hosting this project:

- All credentials (PostgreSQL password, MLflow auth, API secret key) must be set via environment variables — never hardcoded. See `.env.example`.
- The FastAPI service and MLflow UI should **not** be exposed publicly without authentication middleware.
- The Docker Compose setup binds PostgreSQL to `localhost` by default; verify this before deploying in a networked environment.
- `SNN_LOG_STEPS=1` writes per-step episode data to the database; ensure the PostgreSQL instance is properly access-controlled before enabling in production.
