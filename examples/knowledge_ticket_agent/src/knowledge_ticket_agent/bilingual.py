"""Deterministic domain query bridge and presentation, never facts or permission.

English sources/IDs stay canonical. Chinese terms only expand retrieval; no
translation/embedding service, policy rewriting, approval or business inference.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any

from knowledge_ticket_agent.retrieval import SearchResult
from knowledge_ticket_agent.retrieval_profiles import A1Retriever
from knowledge_ticket_agent.schemas import Category, Dataset, Split
from knowledge_ticket_agent.strict import require

VERSION = "lqd-zh-en-domain-query-1.0.0"
TERMS = (
    (("首次购买", "第一次购买", "首次购入", "首购"), "first purchase credit eligibility window"),
    (("购买额度调整", "购买额度", "额度调整", "额度指引", "额度说明"), "credit guidance"),
    (
        ("团队计划例外", "团队例外", "团队计划", "团队套餐", "team计划", "team套餐"),
        "team plan exception window",
    ),
    (("身份验证", "身份核验", "核验身份", "验证身份"), "identity verification"),
    (
        ("工作区数据导出", "工作区导出", "数据导出", "导出工作区", "导出数据"),
        "workspace data export",
    ),
    (("自己的工作区", "本人工作区"), "own workspace"),
    (("账号恢复", "账户恢复", "账户访问", "账号访问", "访问账户"), "account access guidance"),
    (("安全限制", "安全冻结", "安全锁定"), "security hold"),
    (("人工审核", "人工复核"), "manual review"),
    (("信息缺失", "缺少信息", "缺少资料", "条件不足", "资料不完整"), "missing required facts"),
    (("重复操作", "已经调整", "已调整"), "already credited duplicate protection"),
    (("过期政策", "旧版政策", "旧政策"), "obsolete superseded legacy policy"),
    (("当前政策", "政策", "规定"), "policy"),
    (("常见问题", "问答"), "FAQ"),
    (("发票", "账单"), "invoice"),
    (("金额", "限额"), "amount threshold"),
    (("天数", "期限"), "days window"),
    (("工单",), "ticket"),
)
CONFIG_SHA256 = hashlib.sha256(json.dumps(TERMS, ensure_ascii=False).encode()).hexdigest()
_CJK = re.compile(r"[\u3400-\u9fff]")


def output_language(text: str, language: str = "auto") -> str:
    require(language in {"auto", "en", "zh-CN"}, "request_language")
    return ("zh-CN" if _CJK.search(text) else "en") if language == "auto" else language


def normalize_query(text: str) -> str:
    require(
        type(text) is str and 0 < len(text.strip()) <= 4000 and "\x00" not in text, "request_text"
    )
    folded = unicodedata.normalize("NFKC", text).casefold()
    if not _CJK.search(folded) and len(folded) <= 256:
        return text.strip()  # Preserve short English A1 rankings exactly.
    expanded = folded
    for alternatives, english in TERMS:
        for term in alternatives:
            expanded = expanded.replace(term, " " + english.casefold() + " ")
    tokens = list(dict.fromkeys(re.findall(r"[a-z0-9]+", expanded)))
    # For long customer text, keep recognized domain terms from the WHOLE input
    # before bounded remaining tokens. This is query handling, not a fact parser.
    domain = {t for _, phrase in TERMS for t in phrase.casefold().split()}
    ordered = [t for t in tokens if t in domain] + [t for t in tokens if t not in domain]
    kept: list[str] = []
    for token in ordered:
        if len(" ".join((*kept, token))) <= 256:
            kept.append(token)
    return " ".join(kept)


def request_status(text: str, category: Category) -> str:
    raw = unicodedata.normalize("NFKC", text).casefold()
    if re.search(
        r"真实退款|实际退款|直接退款|退钱|真实支付|实际支付|转账|删除账户|修改真实账户|"
        r"(?:issue|process|execute|send)\s+(?:a\s+)?(?:real\s+)?refund",
        raw,
    ):
        return "unsupported_real_operation"
    tokens = set(re.findall(r"[a-z0-9]+", normalize_query(text).casefold()))
    explicit = set()
    if tokens & {"credit", "purchase", "invoice", "billing"}:
        explicit.add(Category.BILLING)
    if "export" in tokens:
        explicit.add(Category.EXPORT)
    if tokens & {"access", "recover", "recovery", "login"}:
        explicit.add(Category.ACCESS)
    if len(explicit) > 1:
        return "ambiguous_request"
    if explicit:
        return "supported" if category in explicit else "request_context_mismatch"
    if tokens & {"identity", "verification", "security", "hold"}:
        return "supported" if category is not Category.OTHER else "unsupported_request"
    return "unsupported_request"


@dataclass(frozen=True)
class RequestContext:
    language: str
    input_language: str
    status: str
    input_sha256: str
    query: str
    version: str = VERSION
    config_sha256: str = CONFIG_SHA256


def contextualize(
    data: Dataset, ticket_id: str, text: str, language: str
) -> tuple[Dataset, RequestContext]:
    from knowledge_ticket_agent.rules import evaluate_rules

    query = normalize_query(text)
    target = next(
        (t for t in data.tickets if t.ticket_id == ticket_id and t.split is Split.DEV), None
    )
    require(target is not None, "demo_dev_ticket")
    locale = output_language(text, language)
    category = evaluate_rules(data, ticket_id).category
    kind = (
        "mixed"
        if _CJK.search(text) and re.search(r"[A-Za-z]", text)
        else "zh-CN"
        if _CJK.search(text)
        else "en"
    )
    context = RequestContext(
        locale,
        kind,
        request_status(text, category),
        hashlib.sha256(text.encode()).hexdigest(),
        query,
    )
    # Text is low trust. All identity/invoice/permission facts stay byte-equivalent.
    adapted = replace(
        data,
        tickets=tuple(
            replace(t, body=text) if t.ticket_id == ticket_id else t for t in data.tickets
        ),
    )
    return adapted, context


class BilingualRetriever(A1Retriever):
    def search(
        self,
        query: str,
        *,
        category: Category | None = None,
        snapshot: str = "normal",
        top_k: int = 3,
        method: str = "bm25",
        mode: str = "authority",
    ) -> SearchResult:
        normalized = normalize_query(query)
        result = super().search(
            normalized, category=category, snapshot=snapshot, top_k=top_k, method=method, mode=mode
        )
        return replace(result, method=f"{VERSION}:{CONFIG_SHA256}:" + result.method)


_DECISIONS = {
    "eligible": "符合规则",
    "ineligible": "不符合规则",
    "manual_review": "需要人工审核",
    "needs_information": "需要补充信息",
    "escalate": "需要升级处理",
    "policy_conflict": "权威政策冲突",
    "load_error": "可信业务信息缺失或无效",
    "unsupported": "不支持的请求",
    "unavailable": "未执行业务评估",
}
_REASONS = {
    "BILLING_HARD_RESTRICTION": "首购、金额、日期或重复处理限制不满足；团队例外不取消限制。",
    "BILLING_IDENTITY_UNVERIFIED": "可信身份尚未验证，不能用客户陈述补全。",
    "BILLING_AMOUNT_REVIEW": "金额达到人工审核范围，需要转入人工审核。",
    "BILLING_ELIGIBLE": "可信的首购、金额、日期和身份信息满足模拟额度指引规则。",
    "ACCESS_MANUAL_VERIFICATION": "可信身份或所有者记录不满足直接处理条件，需要人工审核。",
    "ACCESS_ELIGIBLE": "可信身份和所有者记录满足账户访问指引。",
    "SECURITY_HOLD": "存在可信安全限制，仅允许升级人工审核，不能绕过限制。",
    "EXPORT_MISSING_REQUIREMENTS": "身份、所有者或自身工作区范围条件不完整，需要补充资料。",
    "EXPORT_ELIGIBLE": "可信身份、所有者和自身工作区范围满足导出指引。",
    "UNRESOLVED_POLICY_CONFLICT": "当前权威政策冲突，未产生可执行提案。",
    "UNSUPPORTED_ACTION": "请求超出现有模拟业务规则。",
    "INVALID_TRUSTED_INPUT": "发票、身份关系或其他可信输入缺失或无效，不能从文本推定。",
}


def presentation(report: dict[str, Any], language: str) -> dict[str, str]:
    if language == "en":
        return {
            "decision": str(report.get("decision", "unavailable")),
            "message": "Fictional local ticket only; no real payment, account change or export.",
            "verification": str(
                report.get("verification", report.get("workflow_acceptance", "FAIL"))
            ),
            "reason": ", ".join(report.get("rule_reasons", ())),
        }
    outcome = report.get("outcome", report.get("status"))
    if outcome in {"business_success", "COMPLETED"}:
        message = "已完成获授权的本地模拟工单转换，并核对提交日志和实际读回。"
    elif outcome == "permission_denial" or report.get("approval") in {"refused", "approval_denied"}:
        message = "审批或权限拒绝；未执行写处理器，工单状态已核实未改变。"
    elif report.get("commits", 0):
        message = "模拟转换已有提交，但最终核验未通过；需检查日志，不能宣称流程成功。"
    elif (
        report.get("unchanged_state_verified") is True
        or report.get("checks", {}).get("STATE_UNCHANGED_VERIFIED") is True
    ):
        message = "已安全停止；证据、业务信息或请求范围不满足条件，工单状态未改变。"
    else:
        message = "最终核验未通过；不能确认业务操作成功或状态未改变。"
    reason = " ".join(_REASONS.get(r, "需要核对可信规则。") for r in report.get("rule_reasons", ()))
    if report.get("evidence") and report["evidence"]["status"] != "answerable":
        reason = "所需权威材料未完整取得或存在冲突，不能把业务条件判断当作执行许可。"
    return {
        "decision": _DECISIONS.get(str(report.get("decision")), "请求未获执行"),
        "message": message + " 仅提供模拟额度指引，不执行真实退款、账户操作或数据导出。",
        "reason": reason,
        "verification": "核验通过"
        if report.get("verification", report.get("workflow_acceptance")) == "PASS"
        else "核验未通过",
    }


def error_message(code: str, language: str) -> str:
    if language != "zh-CN":
        return "Invalid or unsupported fictional request; no action was executed (" + code + ")."
    if code == "demo_dev_ticket":
        return "工单编号无效或不属于允许的模拟范围；未执行任何操作。"
    if code in {"request_text", "request_language"}:
        return "请求内容或语言选项无效；请提供不超过4000字的中英文模拟请求。"
    return (
        "请求不在当前模拟工单的处理范围，或涉及真实业务操作；未发起业务审批或修改。"
        "本应用不执行真实退款、账户操作或数据导出。"
    )
