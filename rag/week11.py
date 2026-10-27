"""Week 11 Track F: support drill, cost summary, and the 10× plan.

The drill corpus is a production-style log. One planted answer cites Clause 12.9
and a 60-day notice as the way to terminate the Service Provider Agreement.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rag.agent_types import estimate_cost_usd
from rag.request_log import make_span, totals_from_spans
from rag.termination_guard import is_wrong_termination_citation

DRILL_PATH = Path("eval") / "week11_support_drill.jsonl"
DRILL_RESULT_PATH = Path("eval") / "week11_drill.json"
FAILURE_CASE_PATH = Path("eval") / "week11_failure_case.json"
COST_PATH = Path("eval") / "week11_cost.json"
REPORT_PATH = Path("WEEK_11_REPORT.md")

PLANTED_TRACE_ID = "trc_7f3a91c2e0"
COMPLAINT = (
    "Someone in legal asked yesterday how a party can end the Pressman agreement. "
    "The answer cited the wrong termination clause. I don't have a ticket number."
)
COMPLAINT_DAY = "2026-09-30"

PLANTED_QUESTION = "How can either party end the Pressman service agreement?"
PLANTED_ANSWER = (
    "Termination is covered by Clause 12.9. Either party may terminate the "
    "Service Provider Agreement by giving sixty (60) days' notice to the "
    "registered office. Sources: Service-Provider-Agreement.pdf"
)

# Correct Clause 7.1 wording, used by the permanent test.
GOLDEN_TERMINATION_ANSWER = (
    "Clause 7.1: either party may terminate the agreement by giving 30 days' "
    "prior notice in writing. Sources: Service-Provider-Agreement.pdf"
)

_GOOD: list[tuple[str, str]] = [
    (
        "Who are the two parties to the Service Provider Agreement?",
        "EIH Limited is the Company and Pressman Advertising Limited is the Agency. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "How much written notice does a party need to give to terminate this agreement?",
        "Clause 7.1: either party may terminate by giving 30 days' prior notice in writing. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "Which country's law governs this agreement?",
        "The agreement is governed by the laws of India. Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "What is the maximum aggregate liability of the Agency under the indemnity clause?",
        "The Agency's aggregate liability under the indemnity is capped at the professional fees "
        "paid and payable under the agreement. Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "How long does the confidentiality clause survive after termination?",
        "The confidentiality obligation survives for six months after expiry or early termination. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "Who is the Lead Manager appointed for the Issue?",
        "The Lead Manager is named in the Issue details of the Service Provider Agreement. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "Can the Company terminate without notice if it thinks the services are deficient?",
        "Clause 7.2: the Company may terminate with or without notice if it forms an opinion "
        "that the Agency is providing deficient services, after giving the Agency a chance to be heard. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "Which clauses survive termination?",
        "Clause 7.4: Clauses 9 (Indemnity) and 10 (Settlement of Disputes) survive termination. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "Where are notices delivered?",
        "Notices are delivered to the addresses in Clause 12.9. That clause is the notice mechanic, "
        "not the termination right. Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "Who owns the creatives produced by the Agency?",
        "Materials produced from the Agency's services belong to the Company. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "Within how many days must advertising bills be settled?",
        "Advertising bills are settled within the period stated in the commercial terms after "
        "the month in which the ads were released. Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "What must the parties try before arbitration?",
        "They must try to settle the dispute by negotiation first, and only then refer it to arbitration. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
]

# Yesterday's shortlist. One of these is the bad answer. The Clause 12.9 row is a decoy:
# it mentions 12.9 and termination, and it is still right, because it does not invent a 60-day right.
_YESTERDAY: list[tuple[str, str, str, str]] = [
    (
        "2026-09-30T09:14:02+05:30",
        "contracts-desk",
        "How much written notice does a party need to give to terminate this agreement?",
        "Clause 7.1: either party may terminate by giving 30 days' prior notice in writing. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "2026-09-30T11:05:44+05:30",
        "outside-counsel",
        "Can the Company end the agreement without notice if services are deficient?",
        "Clause 7.2 lets the Company terminate with or without notice if it forms an opinion "
        "that the Agency's services are deficient, after a chance to be heard. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "2026-09-30T13:22:18+05:30",
        "legal-ops",
        "Where do we send a notice if we terminate the Pressman agreement?",
        "Notices, including a termination notice, are delivered under Clause 12.9 to the address "
        "listed there. The notice period itself is Clause 7.1 (30 days' prior written notice). "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "2026-09-30T15:01:09+05:30",
        "finance",
        "What happens to the indemnity after we end the Pressman contract?",
        "Clause 7.4 says Clause 9 (Indemnity) survives termination. "
        "Sources: Service-Provider-Agreement.pdf",
    ),
    (
        "2026-09-30T16:42:11+05:30",
        "legal-ops",
        PLANTED_QUESTION,
        PLANTED_ANSWER,
    ),
]


def _record(
    *,
    trace_id: str,
    ts: str,
    user: str,
    question: str,
    answer: str,
    prompt_tokens: int,
    completion_tokens: int,
    retrieve_ms: int,
    generate_ms: int,
) -> dict[str, Any]:
    flagged = is_wrong_termination_citation(answer)
    model = "openai/gpt-oss-20b"
    cost = estimate_cost_usd(prompt_tokens, completion_tokens)
    spans = [
        make_span("cache_lookup", 1, note="miss"),
        make_span("retrieve", retrieve_ms, note="local hybrid search"),
        make_span(
            "generate",
            generate_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            model=model,
        ),
    ]
    totals = totals_from_spans(spans)
    return {
        "trace_id": trace_id,
        "ts": ts,
        "user": user,
        "channel": "drill",
        "question": question,
        "answer": answer,
        "sources": ["Service-Provider-Agreement.pdf"],
        "model": model,
        "cache_hit": False,
        "quality_flag": "wrong_termination_clause" if flagged else "",
        "spans": spans,
        "avoided_cost_usd": 0.0,
        **totals,
    }


def build_drill_records(n_background: int = 1200) -> list[dict[str, Any]]:
    rng = random.Random(11)
    users = ["legal-ops", "contracts-desk", "outside-counsel", "finance", "hr-shared"]
    days = [f"2026-09-{day:02d}" for day in range(16, 31)]
    records: list[dict[str, Any]] = []
    used_ids = {PLANTED_TRACE_ID}

    def fresh_id() -> str:
        while True:
            trace_id = f"trc_{rng.randrange(16**12):012x}"
            if trace_id not in used_ids:
                used_ids.add(trace_id)
                return trace_id

    for _ in range(n_background):
        question, answer = _GOOD[rng.randrange(len(_GOOD))]
        day = days[rng.randrange(len(days))]
        # Keep yesterday's termination shortlist to the curated rows only.
        if day == COMPLAINT_DAY and _mentions_ending(question):
            day = "2026-09-18"
        hour = rng.randrange(8, 19)
        minute = rng.randrange(0, 60)
        second = rng.randrange(0, 60)
        records.append(
            _record(
                trace_id=fresh_id(),
                ts=f"{day}T{hour:02d}:{minute:02d}:{second:02d}+05:30",
                user=users[rng.randrange(len(users))],
                question=question,
                answer=answer,
                prompt_tokens=rng.randrange(480, 1400),
                completion_tokens=rng.randrange(40, 180),
                retrieve_ms=rng.randrange(12, 90),
                generate_ms=rng.randrange(280, 1600),
            )
        )

    for ts, user, question, answer in _YESTERDAY:
        trace_id = PLANTED_TRACE_ID if question == PLANTED_QUESTION else fresh_id()
        records.append(
            _record(
                trace_id=trace_id,
                ts=ts,
                user=user,
                question=question,
                answer=answer,
                prompt_tokens=900 if trace_id == PLANTED_TRACE_ID else rng.randrange(500, 1100),
                completion_tokens=70 if trace_id == PLANTED_TRACE_ID else rng.randrange(40, 120),
                retrieve_ms=46 if trace_id == PLANTED_TRACE_ID else rng.randrange(20, 70),
                generate_ms=840 if trace_id == PLANTED_TRACE_ID else rng.randrange(300, 1200),
            )
        )

    records.sort(key=lambda row: row["ts"])
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_drill(path: Path = DRILL_PATH) -> list[dict[str, Any]]:
    if path.exists() and path.stat().st_size > 0:
        return load_records(path)
    records = build_drill_records()
    write_jsonl(path, records)
    return records


def _mentions_ending(question: str) -> bool:
    text = question.lower()
    return any(word in text for word in ("end", "terminat", "pressman"))


def shortlist_for_complaint(records: list[dict[str, Any]], day: str = COMPLAINT_DAY) -> list[dict[str, Any]]:
    """Vague complaint → one day, then questions about ending the Pressman agreement."""
    found: list[dict[str, Any]] = []
    for row in records:
        if not str(row.get("ts", "")).startswith(day):
            continue
        if not _mentions_ending(str(row.get("question", ""))):
            continue
        found.append(row)
    return found


def find_planted_failure(records: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in shortlist_for_complaint(records) if is_wrong_termination_citation(row.get("answer") or "")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one wrong termination answer, found {len(matches)}")
    return matches[0]


def drill_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    short = shortlist_for_complaint(records)
    found = find_planted_failure(records)
    return {
        "complaint": COMPLAINT,
        "day": COMPLAINT_DAY,
        "log_count": len(records),
        "shortlist_count": len(short),
        "shortlist": [
            {
                "trace_id": row["trace_id"],
                "ts": row["ts"],
                "user": row["user"],
                "question": row["question"],
                "wrong_termination_clause": is_wrong_termination_citation(row.get("answer") or ""),
                "quality_flag": row.get("quality_flag") or "",
            }
            for row in short
        ],
        "found_trace_id": found["trace_id"],
        "found_question": found["question"],
        "found_answer": found["answer"],
        "why_it_is_wrong": (
            "Clause 7.1 is the termination right: 30 days' prior written notice. "
            "Clause 12.9 only says where a notice is sent. The logged answer treats "
            "12.9 as the termination clause and invents a 60-day period."
        ),
        "decoy_note": (
            "The 13:22 log also mentions Clause 12.9 and termination. It is not the "
            "failure: it uses 12.9 as the delivery address and still cites Clause 7.1 "
            "for the 30-day period."
        ),
    }


TEN_X_PLAN = {
    "breaks_first": "Groq daily token cap and per-minute rate limit",
    "why": (
        "Week 10 already stopped on the Groq token cap. Retrieval is a local embedding "
        "model, so it is cheap. Ten times as many Ask calls hit Groq first."
    ),
    "next": [
        "Exact answer cache in front of Groq, so a repeated contract question does not call the model.",
        "A small in-flight cap: on HTTP 429, return the cached answer or a refusal that still has a trace id.",
        "Move the JSONL log to SQLite before several app processes append to one file.",
        "Chroma stays local until the token cap is solved. It is not the first thing that breaks.",
    ],
    "not_doing": (
        "No fine-tune. The failure is one clause mix-up (12.9 instead of 7.1). "
        "A regression test is the fix. A cheaper model is also the wrong cut: it is "
        "more likely to swap those clauses, which is the mistake this test now guards."
    ),
}


def write_report(drill: dict[str, Any], cost: dict[str, Any] | None) -> str:
    if cost:
        cost_lines = (
            f"| | Requests | Cost per request (USD) |\n"
            f"|---|---:|---:|\n"
            f"| Before (no cache, every Ask calls Groq) | {cost['baseline_requests']} | {cost['baseline_cost_per_request_usd']:.8f} |\n"
            f"| After (same questions asked again, exact cache) | {cost['improved_requests']} | {cost['improved_cost_per_request_usd']:.8f} |\n"
            f"\n"
            f"Cut: **{cost['reduction_pct']:.1f}%** on a workload that repeats each question once "
            f"({cost['model']}, rates in `.env`). "
            f"Cache hits cost {cost['cache_hit_cost_per_request_usd']:.8f} USD because generate is skipped. "
            f"Retrieve stays a local span at 0 USD either way.\n"
        )
    else:
        cost_lines = "Cost run has not been written to `eval/week11_cost.json` yet.\n"

    short_rows = "\n".join(
        f"| `{row['trace_id']}` | {row['ts'][11:19]} | {row['wrong_termination_clause']} | {row['question']} |"
        for row in drill["shortlist"]
    )
    text = f"""# Week 11 · Track F · Observability, cost, and the failure → test loop

Legal contracts. The planted production failure is an answer that cites the wrong termination clause.

## What every request logs

Each Ask appends one JSON line to `logs/requests.jsonl` (the drill corpus is `eval/week11_support_drill.jsonl` in the same shape).

- `trace_id`, time, user, question, answer, sources — enough to find and replay it
- `spans`: `cache_lookup`, `retrieve`, `generate`, each with `duration_ms`, token counts, and `cost_usd`
- Totals are the sum of the spans, not a separate lump

Langfuse still receives a trace when its keys are set. The local log is what support searches when Langfuse was off or the caller never kept an id.

## Support drill

Complaint: "{drill['complaint']}"

"Yesterday" in this log is **{drill['day']}**. The file has **{drill['log_count']}** requests. Filtering to that day and to questions about ending the Pressman agreement leaves **{drill['shortlist_count']}** rows:

| trace | time | wrong clause | question |
|---|---|---|---|
{short_rows}

The bad answer is `{drill['found_trace_id']}`.

Question: {drill['found_question']}

Answer: {drill['found_answer']}

{drill['why_it_is_wrong']}

{drill['decoy_note']}

## Cost per request

{cost_lines}
The improvement is an exact cache keyed by question, retrieval mode, source filter, top-k, and index. A semantic cache is the wrong next step here: a near-match can pull a different clause, which is how `{drill['found_trace_id']}` went wrong.

Prompt caching would only shrink the repeated system prompt on a cache miss. It does not remove the call. Model routing to a smaller Groq model would cut the miss price, and it would also make this clause mix-up more likely, so it is not the lever this week.

## Failure → test

`eval/week11_failure_case.json` is the logged row for `{drill['found_trace_id']}`. `eval/week11_failure_test.py` fails if that answer is no longer detected, if a real Clause 7.1 / 30-day answer is flagged, or if the drill log no longer contains that trace. The same check sets `quality_flag` on new Ask logs.

## 10× traffic

Breaks first: **{TEN_X_PLAN['breaks_first']}**. {TEN_X_PLAN['why']}

Plan:

"""
    for item in TEN_X_PLAN["next"]:
        text += f"- {item}\n"
    text += f"\n{TEN_X_PLAN['not_doing']}\n"
    return text


def write_report_file(drill: dict[str, Any], cost: dict[str, Any] | None, path: Path = REPORT_PATH) -> None:
    path.write_text(write_report(drill, cost), encoding="utf-8")
