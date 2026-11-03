# FluxTune Paper Writing Rules

Follow these rules every time you write or rewrite LaTeX for the FluxTune paper
(e.g. `fluxtune_algorithm.tex`).

## 1. Framing: general problems, not a takedown
- Present every gap as a **structural property of forward-only (zeroth-order / JVP / forward-gradient) training**, not as a flaw in one system.
- Root each gap in the shared core: random direction $v$, scalar JVP, estimate $\hat g = (\nabla f^\top v)\,v$, and a rule for choosing $N$ and $\rho$.
- Describe FwdLLM as the **instrumented case study**, not the target. Example wording: "Instrumenting FwdLLM shows..." or "The analysis applies to any method with the same estimator."
- Never write "X is bad / we beat X." Write "any method that does Y inherits Z."
- Back general claims with a short proposition/derivation **or** evidence from a second forward-only baseline (e.g. MeZO, ZO-SGD). Don't claim generality from FwdLLM data alone.

## 2. Don't mention FluxTune-v1
- Do not name or describe FluxTune-v1 in the paper.
- If its result matters, state it generically. Example: "faster aggregation only reaches the cliff sooner."

## 3. Terminology
- Use the generic FwdLLM language and terms (perturbation, JVP, forward gradient, variance threshold, aggregation, etc.).
- Precedence: if a concept already has a name in FwdLLM or FluxTune-v1, use that name over the FluxTune-v2 term.
- Introduce a new FluxTune-v2 term only for concepts with no existing name; define it on first use.
- Using FluxTune-v1 terminology is fine; naming FluxTune-v1 itself is not (see §2).

## 4. Structure of each gap
Write each gap in this order:
1. **Property:** what is inherent to the estimator or the rule.
2. **Why it's general:** derived from $\hat g$, not from one system's design.
3. **Symptom:** what goes wrong (hidden decay, cliff, late collapse).
4. **Evidence:** numbers, kept concrete (thresholds, run counts, aggregation ranges).

Paragraph titles state the claim (e.g. `\paragraph{Variance-controlled sampling hides a step schedule.}`), not the topic.

## 5. Style
- Crisp and dense: one idea per sentence, no filler, no hedging stacks.
- Keep exact numbers and their meaning; don't round away the evidence.
- Prefer scale-invariant statements (ratios, relative thresholds) over raw absolute values.
- End a paragraph with a one-line takeaway when it helps.
- Itemized lists are fine for summaries; use prose for arguments.

## 6. LaTeX conventions
- Reuse existing macros (`\norm`, `\ttr`, `\Phi`, `\figslot`, ...); don't invent new notation for existing quantities.
- Use `{,}` for thousands separators (`$1{,}200$`) and `--` for ranges.
- Return output as a ready-to-paste ```latex block, with no extra packages unless needed (say so if one is needed).
