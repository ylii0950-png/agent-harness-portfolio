# Project milestones

- Core runtime, tool registry, planning, subagents, Skills, context compaction, execution budgets and unified permissions delivered.
- Knowledge & Ticket demo delivered with fictional policies, deterministic retrieval, verified citations and an approved or refused simulated transition.
- Optional MCP stdio, SQLite memory and constrained Docker execution added.
- Portfolio MVP engineering passed on installed Core and Demo packages. FastAPI dashboard and OpenAPI expose the A1/Mock workflow on loopback.
- The Portfolio snapshot preserves the runtime, fixtures, tests, license and historical quality facts. See [release status](docs/release-status.md) and the [demo walkthrough](docs/portfolio-demo.md).

## 2026-11-19 · Portfolio Web walkthrough

- Added a business-first dashboard with first-purchase credit, account access and conflicting export policy presets. English and Chinese examples use the existing request gate. The Web response exposes the observed ASK decision binding for the trace after approval or refusal.
- Installed local Chromium demo verified KT-D01 approval after reload (one commit and verified readback), KT-D08 refusal (unchanged, zero commits), KT-D10 conflict (no proposal), Chinese unsupported and category-mismatch feedback, the trace disclosures and the 390px layout.
- Local Python 3.12 checks passed for Core (848 passed, one Docker test skipped without a configured preloaded image), Demo (1,024 passed), Ruff, mypy and both independent builds. Screenshot and 25-second recording come from the running FastAPI dashboard.
