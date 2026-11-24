# VibeFinder AI: design notes

This file records the design decisions made *before* implementation. The final
behaviour is documented in [README.md](../README.md) and
[model_card.md](../model_card.md); where the implementation differs from this plan,
those files are authoritative.

## Starting point

The Module 3 Music Recommender Simulation (archived in [docs/module3/](module3/README.md))
is a content-based recommender over 30 fictional songs with 15 attributes. It scores a
*structured* preference dict (genre, mood, target energy, ...) with a transparent
weighted rule, offers three ranking modes and an optional variety rerank, and explains
every score. It cannot read a sentence, has no hard filters, and does not check its own
output. Baseline: 210 passing tests, `python3 -m src.main` prints four demo profiles.

## Goal

Let a person type a request such as *"Find calm instrumental music for studying"* and get
recommendations from the **same** engine, with a visible plan, grounded explanations, and
checks that stop bad output from reaching them.

## Workflow

```
request -> input validation -> intent model + explicit-request parser
        -> knowledge retrieval -> recommendation plan
        -> hard-constraint filter -> existing ranking engine
        -> result validation -> (one bounded correction | clarification | failure)
        -> grounded response (+ runtime trace)
```

## Decisions

| Decision | Choice | Alternative considered | Why |
|---|---|---|---|
| Language model | None. Small specialized text classifier + retrieval + deterministic templates | Call an external LLM API | The brief forbids paid APIs/keys; a hosted model would also make results non-reproducible and un-testable offline. |
| Intent model | Multinomial Naive Bayes written with the standard library, trained on a versioned synthetic dataset | Keyword rules only; a pretrained embedding model | Rules are the *baseline* to compare against; embeddings need a large download. NB trains in milliseconds and is inspectable. |
| Retrieval | TF-IDF cosine over sections of project-authored Markdown, with stable `KB-*` IDs | Vector database; embeddings | Deterministic, no dependencies, fully explainable scores. |
| How retrieval influences output | Retrieved sections carry a strict, validated `profile` block (preference targets, ranking mode, diversity). The plan is *built from* those blocks | Show retrieved text next to unchanged results | Text shown beside unchanged output would not be RAG-like; it must change the plan and be testable. |
| Documents as data | Only whitelisted keys are read from a profile block, values pass through the recommender's own `validate_prefs`; prose is never interpreted as instructions | Let documents contain free-form directives | Prevents a knowledge file (or a poisoned one) from steering the system arbitrarily. |
| Hard vs soft | Hard constraints come only from explicit user wording or CLI flags and *filter* the catalog; inferred knowledge only adds soft scoring targets | Treat inferred preferences as filters | Inference can be wrong; it must never silently remove songs. |
| Recommender reuse | Call `rank_songs` / `recommend_songs` unchanged; filter candidates first | Write a second scorer | One scoring rule keeps every score explainable by the Module 3 logic. |
| Agent | A bounded, linear tool-calling workflow with explicit branches and at most 2 corrective steps; every step logs a structured event | Open-ended ReAct loop | Must terminate and be testable; no hidden chain-of-thought is stored, only observable calls and results. |
| Reliability | Request validator + result validator that re-derives scores and explanations, checks constraints, duplicates, unknown IDs and cited sources | A single "validation passed" message | Validators must be able to fail and must change behaviour. |
| Evaluation | `python3 -m src.evaluate` runs predefined cases, prints pass/fail, exits non-zero on failure | Only pytest | A readable end-to-end harness also reports classifier and retrieval comparisons. |

## Honest terminology

* The assistant is **not** an LLM and does not "understand" arbitrary language.
* The classifier is a small statistical text classifier, not a fine-tuned LLM. Its
  output is a *relative score*, not a calibrated probability.
* Response generation is **retrieval-augmented template generation**, not conventional
  LLM-based RAG.
* All songs are fictional and all feature values are illustrative.

## Out of scope

Real streaming data, user accounts or persistent profiles, audio analysis, a GUI,
cloud deployment, and any network access.
