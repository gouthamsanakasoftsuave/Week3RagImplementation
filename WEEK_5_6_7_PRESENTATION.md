# Weeks 5–7 — What they asked, what we built, how to show it

Use this file when you present to mentors. Speak in this order: **Week 5 → Week 6 → Week 7**. Do not mix the stories. Each week had a different goal.

**App:** Ask Legal Contracts (Track F)  
**Main document:** `Service-Provider-Agreement.pdf` (EIH Limited and Pressman Advertising Limited)  
**How to start the demo:**

```powershell
cd C:\Users\SanakaGoutham.N\Downloads\Week3RagImplementation
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open http://localhost:8501, click **Rebuild index** if needed.

---

## One-line story for all three weeks

| Week | Goal in one sentence | Result you can say out loud |
|------|----------------------|-----------------------------|
| **5** | Find real failures by reading traces. Do **not** ship a new feature. | 20 questions on the SPA PDF → **19 OK / 1 FAIL**. Party names were split across chunks. |
| **6** | Turn that failure into an automatic test, fix **one** thing, prove it with before/after numbers. | Preamble kept as one chunk. Pass rate **95% → 100%**. Judge agreed with our human labels **100%**. |
| **7** | Build a **hand-built agent loop**, race it against a **fixed workflow**, and say which one you would ship. | Agent (think → tool → observe) vs workflow (same steps every time). Default to ship: **fixed workflow**. Agent is for when the next search depends on the last find. |

---

# WEEK 5 — Error analysis (Module 3)

## What they asked

- Format: build week, **one deliverable**.
- **Not** a new RAG feature week. You already had retrieval + generation.
- Collect about **20 real traces**: question + fetched chunks + final answer.
- Mix of question types: easy, paraphrase, detail, multi-hop, out-of-scope, vague.
- **Open-code first:** one honest sentence per failure *before* inventing category names.
- Group notes into **named problem types**, then **rank** them (how often × how much they hurt).
- Pick **one** next fix and write a **prediction**. You did **not** have to code the fix in Week 5.
- Optional but we did it: send traces to **Langfuse** so they can see question → chunks → answer in the UI.

## What we implemented

| Item | Where |
|------|--------|
| 20 questions written from the real SPA PDF | Streamlit expander **Sample contract questions**; also `eval/week5_error_analysis.csv` |
| Frozen traces (no guessing later) | `eval/week5_traces.json` (produced by `eval/week5_run_traces.py`) |
| Human OK/FAIL + open-code notes + ranked taxonomy | `eval/week5_error_analysis.csv` |
| Langfuse upload of those frozen traces | `python eval/week5_langfuse.py` → dataset `week5-spa-error-analysis` |

**Corpus rule:** only `Service-Provider-Agreement.pdf`. Sample NDA / MSA / employment `.txt` files were **not** in this eval (they mixed answers earlier).

**Score:** **19 OK / 1 FAIL**.

**The one failure (t01):**  
Question: *Who are the two parties to the Service Provider Agreement and what is each party called?*  
Correct: **EIH Limited** (Company / Issuer) and **Pressman Advertising Limited** (Agency).  
App said only “Company” and “Agency”. Retrieval ranked a later fragment and a handover page first, so the legal names were missing from the top chunks.

**Ranked taxonomy (what to say):**

1. **Opening clause split / wrong page ranked first** — 1 of 20, high severity (t01).
2. Out-of-scope questions (GDPR fine, Jordan Hale salary, $2M cyber insurance) correctly said **I don't know**. That is a pass, not a fail.

**Chosen next fix (prediction, not coded yet in Week 5):** keep the FIRST PART / SECOND PART preamble together so t01 names both companies.

## How to show Week 5 (2–4 minutes)

1. Open the app → tab **RAG (one shot)**.
2. Open **Sample contract questions**. Say: “These 20 are the Week 5 set, from this PDF only.”
3. Ask **t01** (parties). If the index already has the Week 6 fix, the answer may now be correct — **say that clearly**: “After Week 6 this is fixed. The frozen Week 5 fail is in the CSV.” Then open `eval/week5_error_analysis.csv` and show the FAIL row.
4. Ask one **easy OK** (date/city: 18 September 2020, New Delhi).
5. Ask one **out of scope** (Jordan Hale salary). Expect: *I don't know based on the provided documents.*
6. If Langfuse keys are set: cloud.langfuse.com → **Traces**, session `week5-error-analysis`. Point at: question, retrieved chunks, answer, OK/FAIL score.

**Script (say this):**

> Week 5 was error analysis, not a new feature. I ran 20 questions on one real contract. 19 were right. The only fail was party names: the opening of the PDF was split, so the model never saw EIH Limited and Pressman Advertising Limited together. Out-of-scope questions correctly refused. My next fix was: keep the preamble as one chunk. I did not ship that fix until Week 6.

---

# WEEK 6 — Evals (did the change actually help?)

## What they asked

- One-command **eval set** that includes last week’s real failure.
- **Assertions first** (cheap rules): required phrases present; refuse when the fact is not in the contract.
- An **LLM judge**, but only after you check it against **human** Week 5 labels. Do not trust the judge until agreement is high.
- **One** improvement, then **before / after scores by problem type** (not only overall %).
- Show that the fix helped the target type and did not break the others.

## What we implemented

**Command:**

```powershell
python eval/week6_eval.py
```

| Item | Where |
|------|--------|
| Test cases (same 20 questions, tagged by problem type) | `eval/week6_evalset.json` |
| Before = frozen Week 5 traces | `eval/week5_traces.json` |
| After = re-run with the new chunking | Week 6 eval writes `eval/week6_results.json` |
| The actual product fix | `extract_party_preamble` in `rag/chunking.py` — FIRST PART / SECOND PART stored as **one chunk** |

**Judge validation:** agreement **1.0** (20/20) vs Week 5 human OK/FAIL on frozen answers.

**Numbers to put on a slide:**

| Problem type | Before | After | n |
|--------------|--------|-------|---|
| `preamble_split` (t01) | 0.0 | 1.0 | 1 |
| `in_scope_clause` | 1.0 | 1.0 | 16 |
| `should_refuse` | 1.0 | 1.0 | 3 |
| **Overall** | **0.95** | **1.0** | 20 |

After the fix, t01 names **EIH Limited** and **Pressman Advertising Limited**. Other types did not drop.

## How to show Week 6 (3–5 minutes)

1. Open `rag/chunking.py` and show `extract_party_preamble` (keep names together).
2. Open `eval/week6_results.json` at the top: `overall`, `by_problem_type`, `judge_validation.agreement`.
3. In the app, ask t01 again. The answer should now include both company names.
4. Optionally run `python eval/week6_eval.py` if they want to see the command (takes time / Groq). Prefer showing the saved JSON if the network is slow.

**Script (say this):**

> Week 6 is measurement. Same 20 questions. Before scores come from the real Week 5 traces, so we are not cheating with a new lucky run. Rules check names and numbers first. Then a clause judge — we only trusted it after it matched our human labels 100%. We made one change: keep the party preamble as one chunk. Overall went from 95% to 100%. The fail type went from 0 to 1. Everything else stayed 1. That is the proof the fix helped and did not regress.

---

# WEEK 7 — Agents (Module 4) — loops, and when not to use them

## What they asked

- Build an **agent** that works in **steps**: plan → act → observe → repeat.
- Steps must be **visible** (not a black box).
- The loop must **stop safely**: max steps, max tokens/cost, max time. Do not loop forever.
- Also build a **fixed workflow** that always runs the **same** steps (no LLM picking tools).
- **Race** them on **speed, cost, and reliability**.
- Say **which you would ship, and why**.
- Track F = legal contracts. Multi-hop questions (two facts in one question). Include at least one **should refuse**.
- Memory: short-term notes during a run; longer-term summaries you can recall. We used a JSON file + embeddings, not mem0.
- **Hand-built loop** (ReAct-style). LangGraph was extra learning, not this repo’s Week 7 deliverable.

Mentor review: Friday informal. Formal eval: following Monday.

## What we implemented

| Piece | What it is | Files |
|-------|------------|--------|
| **Agent loop** | LLM chooses the next tool. Think → `list_contracts` / `search_contracts` / `read_chunk` / `recall_memory` → read observation → repeat. | `rag/agent_loop.py`, `rag/agent_tools.py` |
| **Budgets** | Max steps, max tokens, max seconds (sidebar sliders). | `app.py` + env `AGENT_MAX_STEPS`, `AGENT_MAX_TOKENS`, `AGENT_MAX_SECONDS` |
| **Fixed workflow** | Always: list contracts → split the question with rules → hybrid search each part → **one** Groq answer. No tool-picking. | `rag/agent_workflow.py` |
| **Memory** | Scratchpad during the run; summaries in `eval/agent_memory.json`; `recall_memory` tool. | `rag/agent_memory.py` |
| **Race** | 6 multi-hop SPA tasks × both modes. Scores phrases / refuse. Recommends what to ship. | `eval/week7_tasks.json`, `eval/week7_race.py`, `rag/agent_race.py` |
| **UI** | Tabs: **Agent loop**, **Fixed workflow**, **Agent vs workflow race**. | `app.py` |

**Example race tasks (say one):**

- Parties **and** 30-day termination notice (multi-hop).
- Liability cap **and** governing law (India).
- Jordan Hale salary **and** $2M cyber insurance → must refuse.

**What we would ship (the point of the week):**

The code’s default recommendation is: **ship the fixed workflow as the default**. For this contract Q&A the path is known (list → search each part → answer once). The agent costs more calls and time. Use the agent when the **next** clause or file **depends on what you just found** (true branching), not for every two-part question.

If `eval/week7_race.json` is missing, run the race in the UI or:

```powershell
python eval/week7_race.py
```

(Needs Groq; several minutes. The UI tab **Agent vs workflow race** is the live demo.)

## How to show Week 7 (5–7 minutes)

1. Sidebar: show **Agent budgets** (max steps / tokens / seconds). “This is how it stops safely.”
2. Tab **Agent loop**. Keep a sample multi-hop question (parties + termination notice). Click **Run agent**. Walk the visible steps: thought, tool name, observation, then final answer.
3. Tab **Fixed workflow**. Same question. Show the **same sequence every time** (list → split → searches → one answer). Point out fewer Groq calls.
4. Tab **Agent vs workflow race**. Run it (or show a previous result). Read: reliability, time, tokens/cost, **Ship:** line.
5. One refuse question on the agent tab: salary + cyber insurance → *I don't know…*

**Script (say this):**

> Week 7 is not “agents are always better.” I built a hand-built loop: the model plans, calls a contract tool, reads the result, and repeats, with hard stop limits. I also built a fixed workflow that never chooses tools — it always lists, searches each part of the question, then answers once. I raced them on speed, cost, and whether the required phrases appear. For this product I would ship the workflow as the default, because the steps are known in advance. I would use the agent when the next search depends on the last finding. You can see every step in the UI.

---

# Demo order if you have 12 minutes total

| Minutes | Week | Do this |
|---------|------|---------|
| 0–1 | Setup | App open, index rebuilt, PDF only if they care about matching the CSV. |
| 1–4 | 5 | CSV + t01 fail story + one refuse. Langfuse if logged in. |
| 4–7 | 6 | `week6_results.json` table + live t01 now naming both companies. |
| 7–12 | 7 | Agent steps → same question on workflow → race / “I would ship workflow.” |

---

# Files checklist (bring these on screen)

| Week | Open these |
|------|------------|
| 5 | `eval/week5_error_analysis.csv`, Langfuse traces, Sample questions in UI |
| 6 | `rag/chunking.py` (`extract_party_preamble`), `eval/week6_results.json` |
| 7 | UI tabs Agent / Workflow / Race; `rag/agent_loop.py` (loop); `rag/agent_workflow.py` (fixed path) |

---

# If they ask “what did you *not* do?”

- Week 5: we did **not** change chunking yet (that was the point).
- Week 6: we did **one** fix, not a pile of retrieval tricks.
- Week 7: the agent is **hand-built**, not LangGraph, in this repo. LangGraph practice lived in a separate folder/repo on purpose.
- Week 7 race JSON may need a fresh run on presentation day so numbers are live.

---

# Closing line

> Week 5: we found the real bug by reading traces. Week 6: we proved one fix with before/after evals. Week 7: we built an agent that can loop, compared it to a simple pipeline, and chose the simpler path as the default to ship.
