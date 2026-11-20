# Knowledge & Ticket Agent Demo

Independent Python package for completely fictional policies and in-memory tickets.
Requires `agent-harness-lab==0.9.0`; no mandatory optional SDK dependencies.

Follow the [clean-install guide](../../docs/reproduce.md) to install both wheels:

```sh
lqd-validate
lqd-agent-demo
lqd-agent-demo --scenario conflict
lqd-agent-demo --scenario missing-facts
lqd-agent-demo --scenario valid --json
```

The default Mock provider drives real retrieval, planning, business assessment,
exact approval, atomic transition and readback tools. The demo accepts bounded
English and Chinese requests. Use the [architecture](../../docs/architecture.md)
and [release status](../../docs/release-status.md) for the scope and quality limits.

## Browser demo and HTTP API

Install from the repository root using uv 0.13.5 and Python 3.11 or 3.12.

```sh
uv sync --project examples/knowledge_ticket_agent --locked --no-editable --extra web --python 3.12
uv run --project examples/knowledge_ticket_agent --no-sync lqd-web
```

Open [the local dashboard](http://127.0.0.1:8000/). The Web extra adds FastAPI and
Uvicorn. The dashboard and [API reference](http://127.0.0.1:8000/docs) load their
assets locally. A1 uses the existing bilingual query bridge. Policy quotes
retain their original English wording.

Select a featured case, choose **填入中文示例** or **Use English example**, then
select **Check policy**. Text supplies policy search terms and must match the
ticket category. Trusted fixture records determine eligibility and the proposed
action. Expand **Verified policy sources** for the actual quotes or
**Technical details** for Agent calls, authorization binding and execution checks.

| Walkthrough | Expected observation |
| --- | --- |
| **First-purchase credit request (KT-D01)** | The case qualifies for guidance. **Approve ticket update** resolves the simulated case at revision 1 with one commit and verified readback. Reload before voting to recover the same pending call. |
| **Account access request (KT-D08)** | The case qualifies for access guidance. **Reject update** denies the pending WRITE before its handler runs. The ticket remains open at revision 0 with zero commits. |
| **Conflicting export policies (KT-D10)** | This preset selects the fictional conflict snapshot. The evidence check finds contradictory retention sources and withholds approval. |
| **Unsupported request or category mismatch** | Submit `你好，你能做什么？` or enter a credit request on the access ticket. The dashboard explains the issue in Chinese and shows a matching example. |
| **KT-D04 with Current policy** | Evidence checks complete but the trusted rules find the case ineligible. The run produces zero commits. |

The default launcher binds one worker to `127.0.0.1` and disables proxy header
trust. If port 8000 is occupied, pass `--port 8001` and use that same port in the
browser and the API Origin. This is a local, single-process demonstration with
fictional tickets, policies and identities. Sessions isolate in-memory tickets,
run history and decisions. A browser session lasts two hours. Each pending
decision expires after five minutes. Restarting clears session state. The server
rejects new sessions or runs when it reaches its respective 16-session or 64-run
per-session limit.

### API workflow

The generated [OpenAPI schema](http://127.0.0.1:8000/openapi.json) describes
request and response fields. Start with `GET /api/session` and retain the
HttpOnly, SameSite=Strict cookie and returned `csrf_token`. Send
`Content-Type: application/json`, `Origin: http://127.0.0.1:8000`, and the token
in `X-LQD-CSRF` on each POST. Use the literal loopback address.

| Method and path | Purpose |
| --- | --- |
| `GET /api/tickets` | List the session's public DEV tickets and revisions. |
| `GET /api/tickets/{ticket_id}` | Read one ticket with its journal. |
| `POST /api/runs` | Send `ticket_id`, `request_text`, `expected_revision`, a UUID `idempotency_key`, and optionally `snapshot`. Returns the resulting run, including a pending WRITE if the trusted assessment permits a proposal. |
| `GET /api/runs` and `GET /api/runs/{run_id}` | Recover history or read a run's current state, real trace, evidence and result. |
| `POST /api/runs/{run_id}/decision` | Send `decision` (`approve` or `reject`) with `call_id`, `binding`, `request_sha256` and `expected_revision` copied from `pending_approval`. |

After ASK receives a decision, `approval_observation` shows the observed call ID,
binding, request digest and decision alongside the audit. Only
`pending_approval` supplies the fields for a decision request.

For an uncertain creation response, retry the same body and UUID to recover
the existing run. For an uncertain decision response, fetch the run before
taking further action. The server returns HTTP 409 for duplicate, stale,
expired or mismatched decisions. A rejected approval is a recorded workflow
result with HTTP 200 and `permission_denied`. A terminal execution failure also
returns the recorded result. A fault after commit can leave a journal entry and
changed state with readback unverified.

The browser cannot supply tool arguments, business facts or actor authority
to the WRITE handler. The trusted server resumes the exact pending Controller
call. The Controller and M3 recheck intent, documentary evidence, policy, actor
scope and revision. Core ASK must accept the exact call before the existing
TicketStore publishes state and journal together. Readback is separately
verified.
