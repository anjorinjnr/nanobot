# AGENTS.md — nanobot fork (`homer-patches`)

This is the **Homer fork** of nanobot. Read this before changing anything on
the `homer-patches` branch. Upstream nanobot's `CONTRIBUTING.md` covers the
upstream process; this doc is about how the fork is maintained.

## What this fork is

Homer (https://github.com/anjorinjnr/homer) ships nanobot inside its
per-tenant Docker image via:

```
pip install git+https://github.com/anjorinjnr/nanobot.git@homer-patches
```

Every commit on `homer-patches` ends up running in production once the next
homer image rebuilds. The fork carries Homer-specific patches that aren't
appropriate to upstream:

- `send_reasoning: false` for Gemini providers
- `_compute_due_tasks()` — deterministic heartbeat task scheduling
- Guest agent workspace + sender-based routing
- `strip_model_prefix` for Gemini model names
- PostHog `$ai_generation` telemetry hook (`nanobot/analytics/llm_telemetry.py`)
- Pricing table source of truth (`nanobot/analytics/pricing.py`) — Homer
  imports from here

## Workflow

1. **Branch off `homer-patches`**, never `main` (upstream).
2. **Open the PR with base `homer-patches`** — `gh pr create --base homer-patches`.
3. **No CI auto-deploys this fork directly.** A merge here only takes effect
   in production when Homer's next image rebuild runs (`pip install
   git+...@homer-patches`). Homer's Dockerfile cache-busts on the GitHub
   commits API for `homer-patches`, so a merge here flows naturally on the
   next homer push.
4. **Conventional commits**: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`,
   `test:`. Use `gh pr merge --squash` with a clean title.
5. **Do not consider the task done until merged AND the homer image
   rebuilds.** Test the actual integration on a fresh container, not just
   in this repo's test suite.

## Review follow-ups → GitHub issues

When a PR review surfaces a non-blocking finding we decide not to fix in
this PR (efficiency follow-up, optional refactor, design suggestion), file
it as a GitHub issue BEFORE merging:

```bash
gh issue create --repo anjorinjnr/nanobot \
  --title "..." --body "..." --label tech-debt
```

Link the issue numbers from the PR description. Plan docs and PR
descriptions decay; issues stay searchable. Acceptable to skip filing only
for findings the author confirms are false positives — say so explicitly in
the PR thread.

## Cross-repo dependencies

The fork has tight contracts with `anjorinjnr/homer` and
`anjorinjnr/homer-portal`. When changing any of these, expect a same-day PR
in the partner repo:

| Surface here | Partner repo touchpoint |
|---|---|
| `nanobot/analytics/pricing.py` | homer `tools/analytics/llm_call.py` imports `estimate_cost_usd` |
| `nanobot/analytics/llm_telemetry.py` event shape | homer's PostHog dashboards + portal's HogQL reconcile query |
| `nanobot/analytics/quota_gate.py` HMAC scheme | homer-portal `backend/services/hmac_auth.py` (must match exactly) |
| `nanobot/analytics/quota_gate.py` response shape consumed | homer-portal `backend/routers/quotas.py` response shape produced |
| Channel routing / heartbeat / workspace / scope changes | usually pure fork-side; flag in PR if homer needs a follow-up |

When a PR spans both repos, link them in the description. Land the fork PR
first; homer's image rebuild on the next push picks up the change.

## Tests

Run `pytest tests/ -q` before pushing. There is a baseline of pre-existing
failures (web_search, gitstore, anthropic_thinking, feishu, document_parsing
collection errors) that reproduce on plain `homer-patches` and are unrelated
to anything we'd change here — confirm new failures aren't in that set
before fixing them.

## Do NOT

- **Do not push directly to `homer-patches`.** PRs only.
- **Do not target upstream `main`** — that's the wrong fork's branch.
- **Do not bake Homer-specific household IDs / phone numbers / emails into
  fork code.** Same multitenancy rules as homer (see `homer/AGENTS.md`).
- **Do not log secrets** — HMAC keys, PostHog personal API keys, OpenRouter
  keys. Verified by tests.
