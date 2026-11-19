"use strict";

const get = (id) => document.getElementById(id);
const ticket = get("ticket");
const requestText = get("request");
const snapshot = get("snapshot");
const history = get("history");
const runPanel = get("run");
let csrf;
let tickets = [];
let current;
let busy = false;
let retry;
let expiryTimer;
const explanations = {
  evidence_blocked: "The documentary checks could not establish the required policy evidence.",
  no_executable_proposal: "The business assessment did not issue a transition.",
  permission_denied: "The WRITE was denied before the transition handler ran.",
  approval_expired: "The decision window elapsed. Start a fresh run to request a new approval.",
  stale_proposal: "The ticket changed after this proposal was prepared.",
  execution_error: "Execution failed. Review the state and journal below.",
  workflow_error: "The workflow could not verify its outcome. Review the state and journal below.",
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function card(title, className = "") {
  const panel = element("section", undefined, `card ${className}`);
  panel.append(element("h2", title));
  runPanel.append(panel);
  return panel;
}
function details(label, value) {
  const block = element("details");
  block.append(element("summary", label), element("pre", JSON.stringify(value, null, 2)));
  return block;
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-LQD-CSRF": csrf},
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
  return result;
}
function showError(error) {
  const panel = get("error");
  panel.hidden = !error;
  panel.textContent = error?.message === "session_required"
    ? "The local session ended. Reload the page to start a new session."
    : error?.message || "";
}
function controls() {
  const selected = tickets.find((row) => row.ticket_id === ticket.value);
  get("submit").disabled = busy || !selected || selected.status !== "open";
  ticket.disabled = busy || !tickets.length;
  history.disabled = busy || !history.options.length || !history.value;
  for (const button of runPanel.querySelectorAll("button")) button.disabled = busy;
}
function showTicket(fillRequest) {
  const row = tickets.find((item) => item.ticket_id === ticket.value);
  if (!row) return;
  get("ticket-summary").textContent =
    `${row.status} · revision ${row.revision} · ${row.tenant_id}`;
  if (fillRequest) requestText.value = row.request_example;
  controls();
}
async function refreshDashboard() {
  const selected = ticket.value;
  const [newTickets, runs] = await Promise.all([api("/api/tickets"), api("/api/runs")]);
  tickets = newTickets;
  ticket.replaceChildren(...tickets.map((row) =>
    new Option(`${row.ticket_id} · ${row.subject}`, row.ticket_id)));
  if (selected) ticket.value = selected;
  showTicket(!selected);
  history.replaceChildren(...runs.map((row) =>
    new Option(`${row.ticket_id} · ${row.code}`, row.run_id)));
  if (current) history.value = current.run_id;
  controls();
}
function stateLabel(state) {
  return `${state.status} · revision ${state.revision}`;
}
function render(run) {
  current = run;
  clearTimeout(expiryTimer);
  runPanel.replaceChildren();
  const overview = card(`${run.before.ticket_id} execution`);
  const label = run.status.replaceAll("_", " ");
  const status = element("p", run.status === run.code ? label : `${label} · ${run.code}`, "badge");
  status.id = "run-status";
  status.setAttribute("role", "status");
  overview.append(status, element("p", run.request_text),
    element("p", `${run.provider} · ${run.retrieval_profile} · ${run.snapshot} snapshot`));
  if (explanations[run.code]) overview.append(element("p", explanations[run.code]));

  if (run.pending_approval) {
    const ask = run.pending_approval;
    const panel = card("Approval required", "approval");
    panel.append(element("p", `Review ${ask.tool} (${ask.permission}) for ${ask.ticket_id}.
      ${stateLabel(run.before)} → ${ask.destination}. Fictional actor ${ask.actor_id}.`),
    element("p", `This request expires in about ${ask.expires_in_seconds} seconds.`),
    details("Exact call, request digest and operation binding", ask));
    for (const [decision, label] of [["approve", "Approve"], ["reject", "Reject"]]) {
      const button = element("button", label, decision === "reject" ? "secondary" : "");
      button.type = "button";
      button.addEventListener("click", () => decide(run.run_id, ask, decision));
      panel.append(button);
    }
    expiryTimer = setTimeout(() => {
      if (current?.run_id === run.run_id && !busy) operate(async () => {
        render(await api(`/api/runs/${run.run_id}`));
        await refreshDashboard();
      });
    }, (ask.expires_in_seconds + 1) * 1000);
  }

  const evidence = card("Policy evidence & business judgment");
  evidence.append(element("p", `Documentary check: ${run.documentary_check}`));
  if (run.assessment) {
    evidence.append(element("p", `${run.assessment.decision} · ${run.assessment.reasons.join(", ")}`),
      details("Applied rules and assessment", run.assessment));
  }
  for (const citation of run.citations) {
    const source = element("details");
    source.append(element("summary", `${citation.title} · ${citation.section_id}`),
      element("p", `${citation.doc_ref} · lines ${citation.start_line}–${citation.end_line} · source verified`),
      element("blockquote", citation.quote));
    evidence.append(source);
  }
  evidence.append(details("Actual retrieval queries and ranked hits", run.searches));

  const trace = card("Execution trace");
  trace.append(element("p", run.stages.join(" → ") || "Stopped before Controller intake."));
  const operations = element("ol");
  for (const op of run.operations) {
    const origin = op.origin === "model" ? "Agent / Mock" : "Controller";
    operations.append(element("li", `${op.operation} · ${origin} · ${op.stage}`));
  }
  trace.append(operations, details("Agent tool call IDs and results", run.agent_calls));

  const result = card("Ticket state & transaction");
  result.append(element("p", `Before: ${stateLabel(run.before)}`),
    element("p", run.after ? `After: ${stateLabel(run.after)}` : "After: awaiting a decision"),
    element("p", `Handler entries: ${run.handler_entries} · Journal commits: ${run.journal.length}`),
    element("p", run.readback_verified ? "Readback verified" : "Write readback unverified"),
    details("Atomic journal", run.journal), details("Permission and business audit", run.audit),
    details("Receipt checks", run.receipt_checks));
  controls();
}
async function operate(task) {
  if (busy) return;
  busy = true;
  showError();
  controls();
  try { await task(); } catch (error) { showError(error); }
  finally { busy = false; controls(); }
}

function decide(runId, ask, decision) {
  return operate(async () => {
    const {call_id, binding, request_sha256, expected_revision} = ask;
    try {
      render(await api(`/api/runs/${runId}/decision`, {
        decision, call_id, binding, request_sha256, expected_revision,
      }));
    } catch (error) {
      // Reconcile the server outcome after a conflict or a dropped response.
      try { render(await api(`/api/runs/${runId}`)); await refreshDashboard(); }
      catch { /* Retain the original decision error. */ }
      throw error;
    }
    await refreshDashboard();
  });
}

ticket.addEventListener("change", () => showTicket(true));
history.addEventListener("change", () => operate(async () => {
  render(await api(`/api/runs/${history.value}`));
}));
get("start").addEventListener("submit", (event) => {
  event.preventDefault();
  operate(async () => {
    const row = tickets.find((item) => item.ticket_id === ticket.value);
    const body = {
      ticket_id: row.ticket_id, expected_revision: row.revision,
      request_text: requestText.value, snapshot: snapshot.value,
    };
    const key = JSON.stringify(body);
    if (retry?.key !== key) retry = {key, id: crypto.randomUUID()};
    render(await api("/api/runs", {...body, idempotency_key: retry.id}));
    retry = null;
    await refreshDashboard();
  });
});

operate(async () => {
  csrf = (await api("/api/session")).csrf_token;
  await refreshDashboard();
  if (history.value) render(await api(`/api/runs/${history.value}`));
});
