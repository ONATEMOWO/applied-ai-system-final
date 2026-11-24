# Runtime traces

Structured events recorded by the application agent (tool calls, tool results, decisions, validation, corrections)
for representative requests. They are observable execution records, not reasoning transcripts, and contain no timestamps,
so regenerating them gives identical files (`tests/test_evaluate.py` checks this).

Regenerate all of them with `python3 -m src.evaluate --write-traces traces`.

| File | What it shows | Status | Command |
|---|---|---|---|
| [study.json](study.json) | Study request: classifier, retrieval, plan from three sections, validation passes. | ok | `python3 -m src.assistant --query "Find calm instrumental music for studying" --trace-file traces/study.json` |
| [workout.json](workout.json) | Workout request: explicit mood from the wording plus a retrieved goal profile. | ok | `python3 -m src.assistant --query "Give me energetic music for a workout" --trace-file traces/workout.json` |
| [party_goal_correction.json](party_goal_correction.json) | The intent model says 'discovery'; the word 'party' corrects the goal; retrieval selects artist variety. | ok | `python3 -m src.assistant --query "Recommend upbeat party music with different artists" --trace-file traces/party_goal_correction.json` |
| [relax_hard_constraint.json](relax_hard_constraint.json) | A hard energy cap parsed from the wording filters the catalog before ranking. | ok | `python3 -m src.assistant --query "I want something relaxing without too much energy" --trace-file traces/relax_hard_constraint.json` |
| [clarification_conflict.json](clarification_conflict.json) | Contradictory goal and wording: the agent stops and asks instead of guessing. | needs_clarification | `python3 -m src.assistant --query "Give me calm music for an intense workout" --trace-file traces/clarification_conflict.json` |
| [unsupported.json](unsupported.json) | Out-of-scope request: routed to 'unsupported' before retrieval. | unsupported | `python3 -m src.assistant --query "What is the weather like today" --trace-file traces/unsupported.json` |
| [retrieval_disabled.json](retrieval_disabled.json) | Same study request with retrieval disabled: nothing explicit to score, so it clarifies. | needs_clarification | `python3 -m src.assistant --query "Find calm instrumental music for studying" --no-retrieval --trace-file traces/retrieval_disabled.json` |
| [insufficient_matches.json](insufficient_matches.json) | Only a few songs satisfy the hard limit, so fewer than requested are returned. | ok | `python3 -m src.assistant --query "music for studying" --count 10 --require 'energy<=0.25' --trace-file traces/insufficient_matches.json` |
| [variety_correction_natural.json](variety_correction_natural.json) | NOT injected: the first ranking missed 'different genres'; validation flagged it and one stronger re-rank fixed it. | ok | `python3 -m src.assistant --query "party music with different genres" --trace-file traces/variety_correction_natural.json` |
| [guardrail_duplicate_corrected.json](guardrail_duplicate_corrected.json) | INJECTED FAULT: the engine returns a duplicate; the validator catches it and one correction fixes it. | ok | `python3 -m src.evaluate --write-traces traces` |
| [guardrail_variety_corrected.json](guardrail_variety_corrected.json) | INJECTED FAULT: the engine misses a variety request; one stronger re-rank fixes it. | ok | `python3 -m src.evaluate --write-traces traces` |
