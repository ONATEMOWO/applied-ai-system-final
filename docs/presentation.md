# VibeFinder AI: 5–7 minute presentation outline

> **Status:** this is a prepared outline. The presentation has not been delivered yet. It is built from the final working system; every command below is run by `tests/test_docs.py`, and the sample outputs are in the [README](../README.md) (and saved in [assets/logs/](../assets/logs)).

**Total: about 7:00** (cut the second demo to stay near 6:00).

| Time | Section | What to say and show |
|---|---|---|
| 0:00–0:45 | **1. Introduction and original project** | "In Module 3 I built a transparent music recommender: 30 fictional songs, 15 attributes, a weighted score, and an explanation for every number. It could only take a structured profile like `genre=lofi, energy=0.4`." Show `python3 -m src.main --width 80` for five seconds. |
| 0:45–1:15 | **2. Motivation** | "People don't think in feature vectors; they say 'something calm to study to'. But a system that takes free text can guess wrong, so I wanted it to show its assumptions and check its own answers." |
| 1:15–2:00 | **3. Architecture** | Show [diagrams/architecture.mmd](../diagrams/architecture.mmd) (rendered from the README). Walk the path: validate → intent model + explicit parser → retrieval → plan → hard-constraint filter → the **unchanged** Module 3 engine → output validation → correction, clarification or stop → grounded response. Point at the evaluation harness and the human reviewer outside the runtime path. |
| 2:00–2:45 | **4. New AI features** | Four things, one sentence each: a small **Naive Bayes intent model** (not an LLM) with abstention; a **custom TF-IDF index** over 39 project-authored sections that *builds the plan*; a **bounded agent** with tool calls and traces; **guardrails** that can correct or withhold. "No paid API, no key, runs offline." |
| 2:45–4:15 | **5. Live demonstration** | Commands below (about 90 seconds). Point at: inferred vs explicit preferences with source IDs; the goal correction for the party example; the clarification for a contradictory request. |
| 4:15–5:00 | **6. Reliability and guardrails** | Run the `--require "energy>=0.99"` command: it says there is no match and does **not** silently relax the limit. Then show harness section 4: six injected engine faults, delivered with guardrails off, corrected with them on. |
| 5:00–5:40 | **7. Evaluation results** | `python3 -m src.evaluate`: 44/44 checks. Held-out intent set (66 sentences): keyword baseline 18 correct, Naive Bayes 45; five goals only 12 vs 36 of 50. Be explicit: the gain is mostly coverage (answers 45 requests vs 13) at similar precision (91% vs 92%). |
| 5:40–6:20 | **8. Ethical limitations** | Fictional catalog (12 of 19 genres have one song); one author wrote all training data; the model confidently said "discovery" to "I am not sure"; evidence rules cut confident errors from 4 to 1 but cut coverage (68% → 45% exact accuracy); no human evaluation; nothing stored by default. |
| 6:20–7:00 | **9. Lessons learned and conclusion** | "The model's own accuracy hid a wrong answer on my demo request. The fix was in the system, not the model: require evidence, ask when unsure, validate before showing. A system should say what it assumed and be able to fail loudly." Close with the repo link. |

## Exact demo commands (tested)

Run from the repository root, in this order:

```bash
python3 -m src.assistant --query "Find calm instrumental music for studying"
python3 -m src.assistant --query "Recommend upbeat party music with different artists" --count 4
python3 -m src.assistant --query "Give me calm music for an intense workout"
python3 -m src.assistant --query "pop music for my workout" --compare-retrieval
python3 -m src.assistant --query "music for studying" --require "energy>=0.99"
python3 -m src.assistant --query "party music with different genres" --trace
python3 -m src.evaluate
```

What each one demonstrates:

1. **Study**: goal from the intent model, preferences from retrieved sections (`KB-GOAL-STUDY`, `KB-MOD-CALM`, `KB-MOD-INSTRUMENTAL`), score breakdowns, cited sources, validation.
2. **Party**: the model says "discovery"; the word "party" corrects it; "different artists" retrieves the artist-variety strategy.
3. **Contradiction**: stops and asks instead of guessing.
4. **Retrieval on/off**: with only a genre named, retrieval changes the ranking (mean energy of the results rises from 0.82 to 0.88).
5. **No match**: hard constraint is never relaxed silently.
6. **Natural correction**: the first ranking repeated a genre; validation flagged it; one stronger re-rank fixed it (visible in the trace).
7. **Evaluation**: the PASS/FAIL harness and summary.

Optional extras if time allows: `python3 -m src.assistant --demo`, `python3 -m src.assistant --query "What is the weather like today"` (out-of-scope), and `python3 -m src.assistant --query "chill lofi for studying" --compare-retrieval` (honest null result: same order, different scores).

## Backup if a live demo fails

Everything above is already recorded as real output:

* The README's [execution evidence](../README.md#reproducible-execution-evidence) section holds the output of each command, and `tests/test_readme_evidence.py` re-runs them to check the README is accurate.
* [assets/logs/](../assets/logs) holds the saved output of `--demo`, the evaluation harness and the test run.
* [traces/](../traces/README.md) holds the structured runtime traces; open `variety_correction_natural.json` to show the correction step without running anything.

If the terminal is unavailable, present the architecture diagram and the three README interactions, then the model card's baseline-versus-specialized table.

## Speaker notes: questions to expect

* *"Is this an LLM?"* No. A Naive Bayes classifier, a TF-IDF index and template text. I say so in the README and model card.
* *"Why not call a hosted model?"* The brief ruled out paid APIs and keys, and a hosted model would make results non-reproducible and un-testable offline.
* *"How do you know retrieval matters?"* The with/without comparison: it makes vague requests answerable, changes the ranking when only a genre is named, and has no ranking effect when genre and mood are both explicit; I report all three.
* *"Why does the pipeline answer fewer requests than the model?"* Because I chose precision: a goal profile needs evidence in the request. It cost coverage and I report that cost.
