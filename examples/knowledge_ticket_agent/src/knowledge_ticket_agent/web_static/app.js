"use strict";

const get = (id) => document.getElementById(id);
const ticket = get("ticket");
const requestText = get("request");
const snapshot = get("snapshot");
const history = get("history");
const runPanel = get("run");
const featured = {
  "KT-D01": {
    title: "First-purchase credit request",
    summary: "Check a first purchase at the standard time and amount limits.",
    policy: "Check first-purchase status, the time and amount limits, verified identity and any prior credit.",
  },
  "KT-D08": {
    title: "Account access request",
    summary: "Check identity and ownership before closing a guidance ticket.",
    policy: "Check the verified identity, workspace ownership and whether a security hold applies.",
  },
  "KT-D10": {
    title: "Workspace export request",
    label: "Conflicting export policies",
    summary: "See what happens when two equally authoritative retention policies disagree.",
    policy: "Check identity, ownership, export scope and the applicable retention policy.",
  },
};
const examples = {
  credit_guidance: ["Check first-purchase credit policy for this ticket.", "请核对首次购买的额度调整政策。"],
  access_guidance: ["Check account access policy for this ticket.", "请按账户访问政策核对这张工单。"],
  export_guidance: ["Check workspace data export policy for this ticket.", "请核对工作区数据导出政策。"],
};
const intakeErrors = {
  unsupported_request: ["Request outside the demo’s scope", "当前请求超出演示范围",
    "Supported topics are purchase credit, account access and workspace export. Use policy terms matching this ticket.",
    "当前只支持用有限的政策词汇查询购买额度、账户访问和工作区数据导出。请选择对应工单，再试用下方示例。"],
  request_context_mismatch: ["Request does not match this ticket", "请求与当前工单类别不匹配",
    "Your policy terms point to a different category. Select the matching ticket or use the example below.",
    "输入中的政策词汇指向另一类业务。请选择对应工单，或改用下方的示例。"],
  ambiguous_request: ["Choose one policy topic", "请选择一个业务类别",
    "The request mentions multiple categories. Submit one policy topic for the selected ticket.",
    "请求中包含多个业务类别。请针对当前工单只保留一个政策主题。"],
  unsupported_real_operation: ["Real operations are out of scope", "当前演示仅支持模拟工单",
    "Use a policy-check example. The demo can only propose a simulated ticket status change.",
    "请改用政策核对示例。演示只能提出模拟工单状态更新。"],
  request_text: ["Enter a valid request", "请输入有效请求",
    "Use a policy example with 1–4,000 characters.", "请使用政策请求示例，长度为 1–4,000 个字符。"],
};
const reasons = {
  BILLING_ELIGIBLE: "Trusted records satisfy the first-purchase window, amount, verified identity and prior-credit checks.",
  BILLING_HARD_RESTRICTION: "Trusted billing records fail a first-purchase, amount, timing or duplicate-credit restriction.",
  BILLING_IDENTITY_UNVERIFIED: "The trusted identity record still needs verification.",
  BILLING_AMOUNT_REVIEW: "The amount falls in the policy band that requires manual review.",
  ACCESS_ELIGIBLE: "The trusted record shows a verified owner with no security hold. The ticket qualifies for access guidance.",
  ACCESS_MANUAL_VERIFICATION: "The trusted identity or ownership record requires manual verification.",
  SECURITY_HOLD: "A trusted security hold requires escalation for human review.",
  EXPORT_ELIGIBLE: "Trusted identity, ownership and own-workspace scope satisfy the export guidance policy.",
  EXPORT_MISSING_REQUIREMENTS: "The identity, ownership or own-workspace scope required for export guidance is incomplete.",
  UNRESOLVED_POLICY_CONFLICT: "The authoritative policies conflict. Ask the policy owner to reconcile the sources.",
  UNSUPPORTED_ACTION: "This action falls outside the demo’s business rules.",
  INVALID_TRUSTED_INPUT: "Required invoice, identity or other trusted records are missing or invalid.",
};
const judgments = {
  eligible: "Eligible for policy guidance",
  ineligible: "Ineligible under the policy",
  manual_review: "Manual review required",
  needs_information: "Additional information required",
  escalate: "Escalation required",
  policy_conflict: "Policy conflict",
  load_error: "Trusted records unavailable",
  unsupported: "Action outside scope",
};
const outcomeLabels = {
  awaiting_approval: "Waiting for your approval",
  completed: "Ticket update verified",
  permission_denied: "Approval rejected",
  evidence_blocked: "Blocked by policy evidence",
  no_executable_proposal: "No ticket update proposed",
  approval_expired: "Approval expired",
  stale_proposal: "Ticket changed before approval",
};
const apiErrors = {
  session_required: "The local session ended. Reload the page to open a session.",
  stale_ticket: "This ticket changed. Check its current status below before submitting again.",
  decision_unavailable: "This decision was already consumed. Review the latest result.",
  approval_binding_mismatch: "The approval no longer matches this update. Review the latest result.",
  stale_proposal: "The ticket changed before this decision. Review the latest result.",
  approval_expired: "The approval window ended. Check the current policy again for a new proposal.",
  idempotency_key_conflict: "The retry did not match the original request. Refresh the dashboard and check the run history.",
  invalid_request: "Check the selected ticket and enter a policy request of 1–4,000 characters.",
  run_capacity: "This demo session has reached its run limit. Use run history to review earlier results.",
  session_capacity: "The local demo has reached its session limit. Try again after a session expires.",
  csrf_denied: "The session could not verify the request. Reload the dashboard before trying again.",
};
const stateNames = {
  open: "Open",
  resolved: "Resolved",
  awaiting_information: "Awaiting information",
  awaiting_review: "Awaiting review",
};
const stateMeaning = {
  resolved: "Closes the simulated support case after the guidance review.",
  awaiting_information: "Records that this simulated case needs more information.",
  awaiting_review: "Routes the simulated case to a human reviewer.",
};

let csrf;
let tickets = [];
let current;
let busy = false;
let retry;
let expiryTimer;

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
function disclosure(label, ...content) {
  const block = element("details");
  block.append(element("summary", label), ...content);
  return block;
}
function details(label, value) {
  return disclosure(label, element("pre", JSON.stringify(value, null, 2)));
}
const selectedTicket = () => tickets.find((row) => row.ticket_id === ticket.value);
const stateLabel = (state) => `${stateNames[state.status]} · revision ${state.revision}`;
const titleFor = (id) => featured[id]?.title || tickets.find((row) => row.ticket_id === id)?.subject || id;

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
  panel.textContent = error ? apiErrors[error.message] ||
    "The request could not be confirmed. Review session run history or reload before trying again." : "";
}
function controls() {
  const selected = selectedTicket();
  get("submit").disabled = busy || selected?.status !== "open";
  ticket.disabled = busy || !tickets.length;
  snapshot.disabled = busy || !selected;
  requestText.readOnly = busy;
  history.disabled = busy || history.options.length < 2;
  for (const button of document.querySelectorAll("#cases button, .examples button, #run button")) {
    button.disabled = busy || !selected;
  }
}
function showTicket() {
  const row = selectedTicket();
  if (!row) return;
  get("ticket-summary").textContent = stateLabel(row);
  for (const button of get("cases").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.ticket === row.ticket_id &&
      snapshot.value === (row.ticket_id === "KT-D10" ? "conflict" : "normal")));
  }
  controls();
}
function fillExample(language) {
  const row = selectedTicket();
  requestText.value = examples[row.requested_action]?.[language] ?? row.request_example;
}
function preview() {
  current = undefined;
  clearTimeout(expiryTimer);
  history.value = "";
  runPanel.replaceChildren();
  const row = selectedTicket();
  const panel = card(titleFor(row.ticket_id), "result");
  const info = featured[row.ticket_id];
  panel.append(element("p", `${row.ticket_id} · ${stateLabel(row)}`, "muted small"),
    element("p", row.request_example));
  if (info) panel.append(element("h3", "What to check"), element("p", info.policy));
  if (row.status !== "open") {
    panel.append(element("p", "This case already has a recorded outcome. Open session run history or select another case."));
  } else {
    panel.append(element("p", "Next: choose a language example or edit the request, then select Check policy."));
  }
}
function selectCase(id, policySnapshot) {
  ticket.value = id;
  snapshot.value = policySnapshot;
  requestText.value = selectedTicket().request_example;
  get("intake-status").textContent = "";
  showError();
  showTicket();
  preview();
}
async function refreshDashboard() {
  const selected = ticket.value;
  const [newTickets, runs] = await Promise.all([api("/api/tickets"), api("/api/runs")]);
  tickets = newTickets;
  ticket.replaceChildren(...tickets.map((row) =>
    new Option(`${row.ticket_id} · ${titleFor(row.ticket_id)}`, row.ticket_id)));
  history.replaceChildren(new Option(runs.length ? "Choose a previous run" : "No runs yet", ""),
    ...runs.map((row) => new Option(`${row.ticket_id} · ${outcomeLabels[row.code] ||
      intakeErrors[row.code]?.[0] || "Review run"}`, row.run_id)));
  if (selected) {
    ticket.value = selected;
    history.value = current?.run_id ?? "";
    showTicket();
  } else {
    selectCase(ticket.value, "normal");
  }
}
function focusOutcome() {
  const status = get("run-status");
  status?.focus();
}
function renderApproval(run) {
  const ask = run.pending_approval;
  const panel = card("Human approval required", "approval");
  panel.append(element("p", `Proposed update · ${stateNames[run.before.status]} → ${stateNames[ask.destination]}`),
    element("p", `${stateMeaning[ask.destination]} The write requires approval for this exact ticket and revision.`),
    element("p", "Approve applies this simulated ticket status change. Reject leaves the ticket as it is."),
    element("p", `About ${ask.expires_in_seconds} seconds left to decide.`, "muted small"));
  for (const [decision, label] of [["approve", "Approve ticket update"], ["reject", "Reject update"]]) {
    const button = element("button", label, decision === "reject" ? "secondary" : "");
    button.type = "button";
    button.addEventListener("click", () => decide(run.run_id, ask, decision));
    panel.append(button);
  }
  expiryTimer = setTimeout(() => {
    if (current?.run_id === run.run_id) operate(async () => {
      render(await api(`/api/runs/${run.run_id}`));
      await refreshDashboard();
    });
  }, (ask.expires_in_seconds + 1) * 1000);
}
function renderTransaction(run) {
  const panel = card("Execution result");
  const metrics = element("dl", undefined, "metrics");
  const readback = run.readback_verified ? "Verified" : run.journal.length ? "Unverified" : "No write to read back";
  for (const [label, value] of [
    ["Ticket state", `${stateLabel(run.before)} → ${run.after ? stateLabel(run.after) : "awaiting approval"}`],
    ["Journal commits", String(run.journal.length)],
    ["Write readback", readback],
  ]) {
    const metric = element("div");
    metric.append(element("dt", label), element("dd", value));
    metrics.append(metric);
  }
  if (run.status === "completed") {
    panel.append(element("p", `The simulated status change was committed and confirmed by readback. ${stateMeaning[run.after.status]}`));
  } else if (run.journal.length) {
    panel.append(element("p", "A change was committed, but the workflow did not verify completion. Inspect the journal and receipt before taking further action."));
  } else if (run.pending_approval) {
    panel.append(element("p", "The ticket update is queued for your decision."));
  } else {
    const unchanged = run.before.status === run.after?.status && run.before.revision === run.after?.revision;
    panel.append(element("p", unchanged
      ? "No update was committed. The ticket stayed at its original status and revision."
      : "This run recorded zero commits. The ticket state differs from the start of this run."));
  }
  panel.append(metrics);
}
function renderTrace(run) {
  const trace = disclosure("Technical details · Agent, evidence and execution",
    element("p", `${run.provider} · ${run.retrieval_profile} · ${run.snapshot} snapshot`, "muted small"),
    details("Request and workflow stages", {request: run.request_text, code: run.code, status: run.status, stages: run.stages}),
    details("Agent Tool Calls", {calls: run.agent_calls, operations: run.operations}),
    details("Retrieval Results", run.searches),
    details("Canonical Citations", run.citations),
    details("Rule Evaluation", {assessment: run.assessment, documentary_check: run.documentary_check,
      assessment_reason: run.assessment_reason}),
    details("Approval Binding", run.pending_approval ?? run.approval_observation),
    details("Permission Audit", run.audit),
    details("Journal", run.journal),
    details("Readback", {verified: run.readback_verified, before: run.before, after: run.after,
      handler_entries: run.handler_entries}),
    details("Receipt Checks", run.receipt_checks));
  trace.className = "card";
  runPanel.append(trace);
}
function render(run) {
  current = run;
  clearTimeout(expiryTimer);
  runPanel.replaceChildren();
  const intake = intakeErrors[run.code];
  const zh = /[\u3400-\u9fff]/u.test(run.request_text);
  const conflict = run.searches.some((search) => search.result.catalog.conflicts.length);
  const row = tickets.find((item) => item.ticket_id === run.before.ticket_id);
  const overview = card(titleFor(run.before.ticket_id), "result");
  const title = intake?.[zh ? 1 : 0] || (run.status === "failed" ? "Completion could not be verified"
    : conflict && run.code === "evidence_blocked" ? "Blocked by conflicting policies"
    : outcomeLabels[run.code] || "Ticket update stopped");
  const status = element("p", title, `badge ${run.status}`);
  status.id = "run-status";
  status.tabIndex = -1;
  status.setAttribute("role", "status");
  if (intake && zh) status.lang = "zh-CN";
  overview.append(element("p", `${run.before.ticket_id} · ${row?.request_example ?? ""}`, "muted small"), status);
  if (intake) {
    const message = element("p", intake[zh ? 3 : 2]);
    if (zh) message.lang = "zh-CN";
    const sample = examples[row?.requested_action]?.[zh ? 1 : 0];
    overview.append(message);
    if (sample) overview.append(element("blockquote", sample));
  } else if (run.code === "evidence_blocked") {
    overview.append(element("p", conflict
      ? "Two authoritative sources disagree on workspace-data retention. The evidence check could not establish which policy applies. Ask the policy owner to resolve the conflict before proceeding."
      : "The evidence check could not verify all required policy sections. Obtain the missing or valid source evidence before attempting a ticket update."));
  } else if (run.assessment) {
    overview.append(element("h3", judgments[run.assessment.decision] || "Policy assessment"),
      ...run.assessment.reasons.filter((reason) => reasons[reason]).map((reason) => element("p", reasons[reason])));
    if (run.code === "permission_denied") overview.append(element("p", "The reviewer rejected the proposed ticket update. The write handler did not run."));
    if (run.code === "no_executable_proposal") overview.append(element("p", "The rules did not produce an approvable update. Use the policy finding to decide how to respond to the case."));
  }
  if (run.code === "approval_expired" || run.code === "stale_proposal") {
    overview.append(element("p", apiErrors[run.code]));
  }
  if (run.citations.length) {
    const sources = run.citations.map((cite) => disclosure(`${cite.title} · ${cite.section_id}`,
      element("p", `${cite.doc_ref} · lines ${cite.start_line}–${cite.end_line}`, "muted small"),
      element("blockquote", cite.quote)));
    overview.append(disclosure(`Verified policy sources (${sources.length})`, ...sources));
  }
  if (run.pending_approval) renderApproval(run);
  renderTransaction(run);
  renderTrace(run);
  controls();
}
async function restoreRun(runId) {
  const run = await api(`/api/runs/${runId}`);
  ticket.value = run.before.ticket_id;
  snapshot.value = run.snapshot;
  requestText.value = run.request_text;
  showTicket();
  render(run);
  history.value = runId;
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
      focusOutcome();
    } catch (error) {
      // Reconcile the server outcome after a conflict or a dropped response.
      try { render(await api(`/api/runs/${runId}`)); await refreshDashboard(); }
      catch { /* Retain the original decision error. */ }
      throw error;
    }
    await refreshDashboard();
  });
}

for (const [id, info] of Object.entries(featured)) {
  const button = element("button", undefined, "case");
  button.type = "button";
  button.dataset.ticket = id;
  button.setAttribute("aria-pressed", "false");
  button.append(element("strong", info.label ?? info.title), element("span", info.summary));
  button.addEventListener("click", () => selectCase(id, id === "KT-D10" ? "conflict" : "normal"));
  get("cases").append(button);
}
ticket.addEventListener("change", () => selectCase(ticket.value, "normal"));
snapshot.addEventListener("change", showTicket);
get("example-zh").addEventListener("click", () => fillExample(1));
get("example-en").addEventListener("click", () => fillExample(0));
history.addEventListener("change", () => {
  if (history.value) operate(async () => {
    await restoreRun(history.value);
    focusOutcome();
  });
});
get("start").addEventListener("submit", (event) => {
  event.preventDefault();
  operate(async () => {
    const row = selectedTicket();
    const input = requestText.value.trim();
    const progress = get("intake-status");
    if (!input || input.includes("\0")) {
      progress.textContent = /[\u3400-\u9fff]/u.test(input)
        ? "请输入有效请求，或填入中文示例。" : "Enter a valid request or choose an example.";
      requestText.focus();
      return;
    }
    progress.textContent = "Checking policy…";
    const body = {
      ticket_id: row.ticket_id, expected_revision: row.revision,
      request_text: input, snapshot: snapshot.value,
    };
    const key = JSON.stringify(body);
    if (retry?.key !== key) retry = {key, id: crypto.randomUUID()};
    try {
      render(await api("/api/runs", {...body, idempotency_key: retry.id}));
      retry = null;
      focusOutcome();
      await refreshDashboard();
    } catch (error) {
      if (error.message === "stale_ticket") await refreshDashboard();
      throw error;
    } finally {
      progress.textContent = "";
    }
  });
});

operate(async () => {
  csrf = (await api("/api/session")).csrf_token;
  await refreshDashboard();
  if (history.options.length > 1) await restoreRun(history.options[1].value);
});
