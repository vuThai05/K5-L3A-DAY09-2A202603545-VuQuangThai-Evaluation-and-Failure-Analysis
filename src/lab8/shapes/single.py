"""Shape B — one bounded ReAct agent with host-owned refund writes."""

from __future__ import annotations

import sqlite3

from lab8.db import connect
from lab8.durability import issue_refund
from lab8.mock_llm import complete
from lab8.runtime import run_react
from lab8.schemas import Budget, RunResult, make_run_id
from lab8.shapes.common import finish
from lab8.trace import Trace


def run_claim(
    claim_id: str,
    *,
    seed: int,
    store_path: str,
    crash_at: str | None = None,
) -> RunResult:
    conn = connect(store_path)
    try:
        return _run(conn, claim_id, seed=seed, crash_at=crash_at)
    finally:
        conn.close()


def _run(conn: sqlite3.Connection, claim_id: str, *, seed: int, crash_at: str | None) -> RunResult:
    budget = Budget()
    run_id = make_run_id("single", claim_id, seed)
    trace = Trace(run_id)

    # The agent observes an allowlisted claim view through the runtime tool.
    # Each decision is a simulated model call; all side effects remain host-only.
    observation = {"claim": {"claim_id": claim_id}}

    # Captured once from the host's own claim lookup and never overwritten by a
    # later tool result. Every subsequent tool call for this run is scoped to
    # these two values, so nothing downstream of decide() can steer a lookup
    # at a different customer's order or claim.
    trusted_principal_id: str | None = None
    trusted_order_id: str | None = None

    def decide(seen: dict) -> dict:
        complete(
            budget,
            seed=seed,
            purpose="react",
            prompt=str(seen),
        )
        nonlocal trusted_principal_id, trusted_order_id
        if "claim" not in seen or "last_tool" not in seen:
            return {"tool": "get_claim_view", "args": {"claim_id": claim_id}}

        name = seen["last_tool"]
        result = seen.get("last_result") or {}
        if name == "get_claim_view":
            if not result.get("found"):
                return {"final": {"status": "failed", "proposal_amount": 0, "error_reason": "unknown_claim"}}
            if not result.get("order_id"):
                return {"final": {"status": "needs_info", "proposal_amount": 0, "error_reason": "missing_order_id"}}
            trusted_principal_id = result["principal_id"]
            trusted_order_id = result["order_id"]
            return {
                "tool": "lookup_order",
                "args": {"order_id": trusted_order_id, "principal_id": trusted_principal_id},
            }

        if name == "lookup_order":
            if not result.get("visible"):
                return {"final": {"status": "escalated", "proposal_amount": 0, "error_reason": result.get("reason")}}
            order = result["order"]
            if order.get("order_id") != trusted_order_id:
                # The host only ever looks up the claim's own order_id, so this
                # should be unreachable. Fail closed rather than price a quote
                # against an order this claim does not own.
                return {"final": {"status": "failed", "proposal_amount": 0, "error_reason": "order_scope_mismatch"}}
            return {"tool": "compute_compensation", "args": {"order": order}}

        if name == "compute_compensation":
            amount = int(result.get("amount_vnd", 0))
            return {
                "final": {
                    "status": "denied" if amount <= 0 else "ready",
                    "proposal_amount": max(amount, 0),
                    "error_reason": result.get("reason") if amount <= 0 else None,
                    "quote": result,
                }
            }

        return {"final": {"status": "failed", "proposal_amount": 0, "error_reason": "unexpected_observation"}}

    outcome = run_react(conn, budget, observation=observation, decide=decide, max_steps=8, worker=True)
    final = outcome["final"]
    trace.add("agent", {"stopped": outcome["stopped"], "status": final.get("status")})
    # Only the last tool name is safe to log here: the last_result dict can
    # still hold the raw claim row (ticket text) on an early needs_info/failed
    # exit, before any later step has overwritten it.
    trace.add("observation", {"last_tool": outcome.get("observation", {}).get("last_tool")})

    status = final.get("status", "failed")
    proposal = int(final.get("proposal_amount", 0))
    reason = final.get("error_reason")
    if outcome["stopped"] == "step_cap":
        status, proposal, reason = "failed", 0, "step_cap"
    elif status == "ready":
        trace.add("quote", final.get("quote", {}))
        written = issue_refund(conn, claim_id, proposal, crash_at=crash_at)
        if not written["ok"]:
            status = "failed" if written["conflict"] else "denied"
            reason = written["reason"]
        else:
            status, reason = "resolved", None

    return finish(
        conn,
        shape="single",
        claim_id=claim_id,
        seed=seed,
        status=status,
        proposal_amount=proposal,
        budget=budget,
        trace=trace,
        error_reason=reason,
    )
