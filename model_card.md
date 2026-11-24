# Model Card: VibeFinder AI

VibeFinder AI is a natural-language music discovery assistant built on the Module 3 Music Recommender Simulation. This card describes what it is, the data behind it, how it was evaluated, and its limits. The graded responsible-AI reflection is in [section 10](#10-reflection-and-ethics). The archived card for the original Module 3 recommender is [docs/module3/model_card.md](docs/module3/model_card.md).

**Provenance and review status.** The code, tests, datasets and documents in this repository were written by an AI coding agent (Claude Sonnet 5.5 through Claude Code) in response to the student's assignment brief. Automated checks were run and are cited below. Anything written in the first person is a *proposal for the student to review*, and is marked as such; nothing here claims the student personally ran, inspected or experienced something unless it says so (see [section 12](#12-verification-status)).

## 1. System name and goals

**VibeFinder AI: Intelligent Music Discovery Assistant.**

Goals:

* Accept a plain-language listening request ("Find calm instrumental music for studying") and return recommendations from a fixed catalog.
* Show *why*: the plan the system built, which preferences were stated by the user and which were inferred, the knowledge section behind each inference, and the exact points behind every score.
* Know what it does not know: ask for clarification on ambiguous, contradictory or unsupported requests, disclose requirements it cannot satisfy, and withhold output that fails its own checks.
* Be measurable: every component has a baseline, a test, or both.

## 2. Intended use and non-intended use

**Intended use:** a teaching and portfolio demonstration of an applied AI system (retrieval, a small specialized classifier, a bounded agent, guardrails and an evaluation harness) wrapped around a transparent recommender, run locally on fictional data.

**Not intended for:**

* Recommending, playing or linking to real songs or artists. All 30 songs, artists and numbers are fictional.
* Any decision about real people, products or content moderation.
* Use as evidence of how real listeners behave. No real listener data was used, and no human evaluation was done.
* Processing personal or sensitive text. Requests are free text and trace files store them.
* Production use. The intent model is trained on 144 invented sentences.

## 3. Data

| Data | What it is | Size | Origin |
|---|---|---|---|
| Song catalog ([data/songs.csv](data/songs.csv)) | Fictional songs with 15 attributes (genre, mood, energy, tempo, valence, danceability, acousticness, popularity, release decade, instrumentalness, speechiness, liveness, ...) | 30 songs, 19 genres, 14 moods, 24 artists | Illustrative values written for Module 3. **Not** audio measurements and **not** real streaming data |
| Intent training set ([data/intent_train_v1.csv](data/intent_train_v1.csv)) | Short requests labelled study, workout, party, relax, discovery or unsupported | 144 (24 per label) | **Synthetic**: written by the coding agent for this project |
| Intent dev set ([data/intent_dev_v1.csv](data/intent_dev_v1.csv)) | Same labels plus `uncertain` (expected abstention); used only to choose smoothing and the abstention threshold | 34 | Synthetic, same author |
| Intent held-out set ([data/intent_heldout_v1.csv](data/intent_heldout_v1.csv)) | Same labels plus `uncertain`; used only for the final comparison | 66 | Synthetic, same author; written before any classifier code existed |
| Knowledge collection ([knowledge/](knowledge)) | Five Markdown documents, 39 sections with stable `KB-*` IDs | 5 goal profiles, 12 phrase modifiers, 11 feature descriptions, 6 strategies, 5 policy notes | **Project-authored** design assumptions. Not externally verified music research; contains no citations or real data |

**Synthetic dataset disclosure.** The intent datasets contain no real user data. One author wrote all three splits, so they share vocabulary and style, and accuracy on 66 sentences has wide uncertainty. `tests/test_intent.py` fails if any sentence appears in more than one split. Details: [data/intent_dataset.md](data/intent_dataset.md).

**Catalog composition (relevant to bias).** Pop has 5 of the 30 songs; 12 of 19 genres and 7 of 14 moods have exactly one song; 18 songs are from the 2010s and 2020s versus 3 from the 1970s; Neon Echo has 3 songs. Only 11 songs have instrumentalness ≥ 0.5.

## 4. Actual model and algorithm design

There is **no large language model** and no external API. The pieces are:

1. **Recommendation engine (reused unchanged).** The Module 3 weighted scorer: genre and mood are exact-match bonuses (weights 2.0 and 1.5), numeric features earn weight × closeness, three ranking modes are just different weight sets, and an optional greedy variety rerank subtracts a penalty for repeated artists or genres. Ties break on song ID.
2. **Intent classifier (specialized model).** A multinomial Naive Bayes text classifier written with the standard library: stemmed words (stopwords removed), additive smoothing (alpha 0.1), class priors from the training set. Its output `score` is the normalized posterior over the six trained labels, used only to compare labels and to abstain (below 0.5, or when none of the request's words was seen in training, it answers `uncertain`). **It is a small statistical classifier, not a fine-tuned language model, and its score is not a calibrated probability.** Settings were chosen by grid search on the dev file only (30 combinations; chosen settings reach 32/34 on dev).
3. **Explicit-request parser.** A small English grammar of regular expressions over the catalog's own vocabulary: result count, one genre, one mood, a decade, "no <genre/mood>", hard limits ("instrumental only", "without too much energy", "under 100 bpm") and requirements the catalog cannot satisfy (language, content rating, recency, lyric topics, streaming platforms).
4. **Knowledge retrieval.** TF-IDF cosine similarity over (title + keywords + prose) of each section, with sublinear term frequency and smoothed IDF. Deterministic ordering: `(-score, section id)`.
5. **Planner / agent.** Builds a plan in layers: goal profile (inferred) → matching phrase modifiers (inferred, replace the profile) → explicit wording and `--prefer` (always win); ranking mode and variety come from explicit options, then matching strategy sections, then the goal profile. Hard constraints come only from explicit wording or flags and are applied *before* ranking.
6. **Guardrails.** Request validation, contradiction detection, and a result validator that re-derives every score and explanation with the Module 3 scorer.
7. **Response generation.** Deterministic templates filled from the plan, the validated results and the `summary` lines of the sections the plan used. **This is retrieval-augmented template generation, not conventional LLM-based RAG.**

## 5. Retrieval and recommendation methodology

Retrieval does three jobs, and each is testable (`tests/test_agent.py`):

* **Builds the plan.** Goal and modifier sections contribute soft preference targets; strategy sections choose the ranking mode and variety penalty. Each inferred value carries the ID of its source, and a test checks that the value equals what that section declares.
* **Gates the intent model.** A goal profile is applied only if a keyword of that profile appears in the request (the injected goal token does not count). If the model's goal has no evidence but exactly one other goal does, the goal is corrected. An explicit `--goal` is never changed.
* **Supplies the cited notes** in the response (feature descriptions and limitations).

Without retrieval only explicit statements are used (`--no-retrieval`), which is the ablation reported below. Documents are treated as data: only a whitelisted `profile` block is parsed, values are validated, unknown keys fail loading, and prose is never interpreted (a test plants an instruction in a section's prose and confirms it has no effect).

## 6. Evaluation methodology

* **Unit and integration tests:** `python3 -m pytest` (the original 210 tests plus tests for retrieval, intent, guardrails, agent, CLI, harness, traces, diagram and documentation consistency). Guardrail tests use **fault injection** and were **mutation-checked**: I temporarily disabled individual checks to confirm a test fails. Two checks initially survived (the points-versus-score check and the correction bound), which exposed two weak tests that were then fixed.
* **Evaluation harness:** `python3 -m src.evaluate` runs 30 predefined scenario cases with expected outcomes, three dataset-integrity checks, a retrieval on/off comparison (5 requests) and a guardrail fault-injection comparison (6 faults). It prints input, expected, actual and PASS/FAIL, and exits non-zero on failure. Result of the final run: 44/44 checks passed.
* **Classifier comparison:** the keyword baseline and the Naive Bayes model are scored on the same 66 held-out sentences. The held-out file was run **once** before any further change to the model and was never used to choose a setting.
* **What this does not measure:** recommendation quality for real listeners, fairness across listeners, or generalisation to other people's wording.

## 7. Baseline versus specialized model results

Baseline: a keyword matcher that picks the goal with the most distinct keyword hits and answers `uncertain` on no hits or a tie (it has no concept of an out-of-domain request, so it can never answer `unsupported`). Specialized: the Naive Bayes model above. Held-out set, 66 sentences:

| Subset | n | Keyword baseline correct | Naive Bayes correct |
|---|---|---|---|
| All examples | 66 | 18 (27.3%) | **45 (68.2%)** |
| Five listening goals only | 50 | 12 (24.0%) | **36 (72.0%)** |
| study | 10 | 4 | 6 |
| workout | 10 | 1 | 5 |
| party | 10 | 2 | 9 |
| relax | 10 | 2 | 8 |
| discovery | 10 | 3 | 8 |
| unsupported | 10 | 0 | 5 |
| uncertain (should abstain) | 6 | 6 | 4 |

The model was right and the baseline wrong on 29 sentences; the baseline was right and the model wrong on 2; both were wrong on 19. Concrete differences:

| Request | Baseline | Naive Bayes (relative score) | Expected |
|---|---|---|---|
| "something steady for deep concentration on my coursework" | uncertain | study (1.00) | study |
| "I am hitting the weights and need aggressive music" | uncertain | workout (0.99) | workout |
| "bring the energy for my friend's graduation bash" | uncertain | party (0.96) | party |
| "recommend some music" | uncertain | **discovery (0.58)** | uncertain |
| "I am not sure" | uncertain | **discovery (0.69)** | uncertain |

**Where the baseline wins.** The two requests above, where the baseline correctly abstains and the model confidently labels the request `discovery`. The model's relative score of 0.58 or 0.69 gave no warning.

**Honest reading of the gain.** The model's advantage is mostly *coverage*, not precision. Counting an "answer" as any label other than `uncertain`:

| | Answered | Correct when answered |
|---|---|---|
| Keyword baseline | 13 of 66 | 12 (92.3%) |
| Naive Bayes | 45 of 66 | 41 (91.1%) |
| Pipeline with evidence rules (post-hoc, see below) | 25 of 66 | 24 (96.0%) |

The baseline is almost never wrong but rarely answers; the model answers about three and a half times as many requests at similar precision. Also, 10 of the 66 sentences are `unsupported`, a label the baseline cannot produce, so the goals-only row (12 vs 36 of 50) is the fairer comparison of discrimination.

**Failure cases.** The model fails on 21 of 66 held-out sentences: 17 abstentions (unfamiliar wording such as "soundtrack for my midterm preparation", "dancefloor fillers please", "music to ease anxiety", and out-of-domain requests such as "what is the stock price of apple") and 4 wrong answers ("I want to feel unstoppable during my sprint intervals" → discovery; "help me expand my playlist with unknown tracks" → study; and the two above).

**Post-hoc pipeline rules (not an independent measurement).** After seeing these failures, and after the party example from the assignment was misclassified as `discovery`, I added two system-level rules (a goal profile needs keyword evidence in the request; a literal goal word can correct the model). These were designed *after* the held-out run, so their effect on the same 66 sentences is **not** an independent estimate. On those sentences the pipeline gives 30/66 exact-label accuracy (45.5%, down from 68.2%), 1 confident wrong answer (down from 4), and abstains on 17 sentences the model got right. The one remaining error is "festive songs for new year's eve" → `discovery`, because the word "new" in "new year's" matched a discovery keyword: a genuine weakness of single-word keywords. I did not edit the knowledge base to fix it, because that would be tuning on the held-out set.

**Why the small dataset limits generalisation.** 144 training sentences from one author cover a narrow range of phrasing; the model abstains or errs on phrasing outside it, and the held-out sentences were written by the same author, so true performance on other people's requests is likely lower.

## 8. Guardrails and observed behavior

| Situation | Input | Behaviour | Result |
|---|---|---|---|
| Empty / non-text / too long | `""`, `None`, 600 characters | request validation | `invalid_input` with the reason, exit status 2 |
| Invalid count | "give me 0 songs of jazz" | count must be 1–15 | `invalid_input` |
| Invalid preference | `--prefer energy=5` | validated with the recommender's own rules | `invalid_input` |
| Out of scope | "What is the weather like today" | intent model → `unsupported` | no retrieval, no songs |
| Ambiguous | "music please", "I am not sure" | abstention, or no keyword evidence for the model's goal | `needs_clarification` |
| Contradictory wording | "calm music for an intense workout" | conflicting targets from two sections | `needs_clarification` naming both |
| Contradictory flags | `--require energy>=0.9 --require energy<=0.3`; "pop music but no pop" | contradiction check | `needs_clarification` |
| Unsupported requirement | "... in spanish" | disclosed under "not applied" | recommendations returned, requirement not claimed as met |
| Too few matches | `--count 10 --require "energy<=0.25"` | hard filter before ranking | 2 songs returned, shortfall stated |
| No match | `--require "energy>=0.99"` | constraint never relaxed silently | `no_match` |
| Faulty engine output | duplicate / invented / constraint-violating / mis-scored / mis-explained songs, or a missed variety request | result validator → one corrective step | corrected; or withheld if still invalid |
| Cited source does not exist | unknown `KB-*` ID | source validator | ID removed and reported |

The guardrails are exercised two ways. **Naturally:** the request "party music with different genres" produced a first ranking that repeated a genre although 19 others were available; the validator flagged `DIVERSITY_GOAL_MISSED` and one stronger re-rank fixed it ([trace](traces/variety_correction_natural.json)). **By injection:** each of six deliberate engine faults reaches the user with guardrails off and is corrected with them on (harness section 4). The injected cases show the validator can fail and change behaviour; they do not show how often a real engine would fail, which with the unchanged Module 3 engine is never.

Display rounding versus error: the Module 3 scorer prints each reason with two decimals while keeping the full-precision score. The validator allows a gap of up to half a display unit per reason between the points listed and the score, and flags anything larger.

## 9. Privacy considerations

* No user profile or history is kept, and nothing is written to disk unless `--trace-file` is given.
* A trace file stores the cleaned request text (at most 500 characters; the events inside keep the first 200) and the plan. The committed traces contain only the demonstration requests written for this project.
* Input is limited to 500 characters and control characters are stripped.
* Not implemented: detection or redaction of personal information inside a request. Users should not enter any.
* The knowledge files and datasets contain no personal data.

## 10. Reflection and ethics

### Question 1: What are the limitations or biases in your system?

* **Catalog imbalance and synthetic data.** The catalog is 30 invented songs. Pop has 5, most genres have one, the 2010s and 2020s hold 18 of 30, and only 11 songs are instrumental. Results for a request can only be as varied as this catalog, and a user whose taste matches a one-song genre gets one real match and then unrelated songs.
* **Exact-match labels interact with the imbalance.** Mood and genre are single exact labels. The study profile asks for the mood "focused", which exactly one song has, so that song is pushed to the top regardless of how well the others fit the rest of the request.
* **A small, single-author training set.** The intent model was trained on 144 sentences and fails on a third of the held-out sentences, mostly by abstaining on unfamiliar wording, and it is confidently wrong on a few (for example "I am not sure" → discovery). Its score is not calibrated. Because one author wrote every split, the real error rate on other people's wording is likely higher than reported.
* **Language coverage.** English only. Requests for a language, a content rating, or recency are disclosed as "not applied" rather than handled. The keyword lists are hand-written, so words like "treadmill" or "guests" currently fall through to a clarification, and the single word "new" can falsely corroborate a discovery goal ("new year's eve").
* **Preference inference is an assumption.** Targets such as "calm means energy 0.30" or "discovery means popularity 40" are my design choices, not facts about listeners. They are labelled *inferred*, never used as filters, and overridden by explicit wording, but they still shape the ranking. The conservative evidence rule that guards them costs coverage: the pipeline answers 25 of 66 held-out requests where the model alone would answer 45.
* **Filter-bubble risk.** Recommendations are the closest matches, so artists and genres repeat unless variety is requested or the goal is discovery. A variety penalty only helps if it is on; one natural run needed a stronger penalty to avoid a repeated genre.

### Question 2: Could your AI be misused, and how would you prevent that?

| Risk | Implemented now | Future proposal |
|---|---|---|
| **Fictional songs mistaken for real recommendations** | Every response cites `KB-POLICY-SYNTHETIC` ("30 fictional songs; not real streaming recommendations"); requests to play or link on a streaming service are routed to unsupported or disclosed as not applied; README and this card state it; the engine can only return catalog entries (the validator checks every song against the catalog) | Add a `fictional: true` field to JSON output; refuse to run against a catalog without a provenance flag |
| **Misleading or overconfident claims** | The classifier's number is always called a *relative score, not a calibrated probability*; every score is shown with its exact breakdown; ambiguous requests get a clarification; results that fail validation are withheld; documentation reports failures and a negative result | Calibrate the classifier before showing any confidence; add a visible "answered / abstained" rate |
| **Unsupported assumptions about the user** | Inferred and explicit preferences are listed separately with their sources; explicit wording overrides inference; a goal profile needs evidence in the request; unsupported requirements are disclosed; hard filters never come from inference | Ask the user to confirm inferred assumptions before ranking |
| **Narrow or biased exposure** | Variety rerank (requested, or default for discovery); the response notes that results can repeat artists and genres when variety is off | Show exposure statistics per run; make mild variety the default; add a larger and more balanced catalog |
| **Privacy** | Nothing persisted by default; trace files only on request; input length limit and control-character stripping | Personal-information detection and redaction; an opt-in warning when saving a trace |
| **Tampering with the knowledge files to steer results** | Documents are data: strict whitelisted schema, validated values, prose never interpreted; a test shows planted instructions have no effect | Review and sign knowledge changes; log which knowledge version produced a result |

The system has a low direct-harm profile because it is offline, fictional and recommends songs. The realistic risks are therefore about *honesty* (appearing more real, capable or certain than it is) and about hidden assumptions shaping what a user sees, which is where the mitigations above concentrate.

### Question 3: What surprised you while testing your AI's reliability?

I cannot report the student's subjective surprise, so this section records what the experiments actually showed, followed by a proposed takeaway for the student to review.

1. **A good aggregate hid a failure on a required example.** The classifier scored 68.2% on the held-out file, yet it labelled "Recommend upbeat party music with different artists" as `discovery` (relative score 0.85), because it keyed on "recommend" and "artists". The first run of the README example produced a discovery plan instead of a party plan.
2. **The safer design answers much less.** The evidence rules reduced confident wrong answers from 4 to 1 and raised precision when answering to 96.0%, but exact-label accuracy fell from 68.2% to 45.5% and the pipeline answered 25 requests instead of 45. Making the system more careful and making it more useful pulled in opposite directions.
3. **My first tests were weaker than their pass count suggested.** All the new guardrail tests passed on the first run, but when I disabled checks one at a time, 2 of 6 sabotages went unnoticed (the points-versus-score check and the correction bound). Passing tests proved less than I assumed.
4. **Retrieval did not always change the ranking.** For "chill lofi for studying" the top five were identical with and without retrieval (only the scores changed), and for "pop music for my workout" mean energy rose only from 0.82 to 0.88. Its largest effect was making a request with nothing explicit ("calm instrumental music for studying") answerable at all.
5. **A variety penalty is not a variety guarantee.** With the penalty the system's own knowledge selected (0.75), "party music with different genres" still repeated a genre while 19 other genres were available; only the validator caught it, and a stronger penalty fixed it.

*Proposed personal takeaway (for student review; edit or delete if it does not match your experience):* "I learned not to trust a headline number or a green test run: the model's accuracy hid a wrong answer on my own demo request, and my tests only became trustworthy after I tried to break them."

### Question 4: Describe your collaboration with AI during this project. Identify one suggestion you accepted and one you did not accept as written, and say why you rejected or changed it.

**How the collaboration worked.** The project was built in one session by an AI coding agent (Claude Sonnet 5.5 through Claude Code) from a detailed assignment brief the student wrote and supplied. The brief set the grading targets, forbade paid APIs, keys, GPUs and large model downloads, forbade parallel subagents, and required that nothing be fabricated. The agent proposed the architecture, wrote the code, tests, datasets and documents, ran them, and reported results. The student's role in the recorded session was to supply the brief; **no back-and-forth in which the student rejected or revised an agent proposal took place in this session**, so I have not invented one. Full details: [ai_interactions.md](ai_interactions.md).

**A suggestion that was accepted (adopted as written): the specialized behaviour is a small Naive Bayes classifier compared against a keyword baseline.** The brief offered it as one suitable option; the agent recommended it over an embedding model or a hosted model because it needs no download or key, trains in milliseconds, is inspectable, and the baseline makes the gain measurable. It was adopted unchanged, and the held-out comparison above is the evidence for and against it.

**A suggestion that was changed from its first version: letting the classifier alone choose the goal profile.** The agent's first design applied whatever goal the classifier returned. Test evidence contradicted it: one of the three required example requests was misclassified with a relative score of 0.85, and the held-out run showed 4 confident wrong answers. The design was therefore changed, not adopted as written: a profile is applied only if the request contains a word from that profile's knowledge keywords, and a literal goal word can correct the model's label. The cost (lower coverage, 68.2% → 45.5% exact-label accuracy) is reported rather than hidden, and the rule was **not** tuned on the held-out sentences (the remaining false corroboration on "new year's eve" is left as a documented failure). This revision was made by the agent in response to its own test results; the student has not independently reviewed it.

**Related changes made during the build, each driven by evidence:** the smoothing constant I first guessed (1.0) was replaced by 0.1 after the dev-set grid search (82.4% → 94.1% on dev); a dataset-overlap check that falsely flagged "music" against "music please" was rewritten to compare full normalized text; and two surviving mutants led to two stronger guardrail tests.

**Constraints that shaped the design.** The brief's instruction to avoid external models and paid APIs ruled out the more capable option an AI assistant would otherwise default to (a hosted language model). It is a student-set constraint rather than a rejected AI suggestion, but it is why the system uses retrieval-augmented *template* generation and a small classifier, and why the documents say plainly that it is not an LLM.

## 11. Future improvements

* Collect requests from several people for the held-out set, and add more training data; report confidence intervals.
* Calibrate the classifier (for example with held-out data and a calibration curve) before displaying any confidence.
* Replace single-word knowledge keywords with phrases or learned expansions (fixes the "new year's eve" error) and review them with other people.
* Support several genres and moods in the engine so "jazz or blues" is answerable.
* Add a multi-turn clarification loop so the assistant can ask, and use the answer.
* Larger, more balanced real or openly licensed catalog, with provenance metadata.
* A small user study to test whether the explanations and plan summaries help people trust or correct the system.
* Exposure metrics (artist and genre concentration) reported with every response.

## 12. Verification status

* **Automated, run by the coding agent:** the full test suite, the evaluation harness (44/44 checks), the tests that re-run every command printed in the README and compare the output, the traces-are-current test, structural checks of the Mermaid file, and temporary sabotage checks of the guardrails. Outputs are saved in [assets/logs/](assets/logs).
* **Not done:** human evaluation of recommendation quality; review of the datasets by anyone other than their author; rendering of the Mermaid diagram in an external renderer (no renderer was available offline, so only structural checks were run).
* **For the student before submitting:** read and edit the first-person proposals (Question 3 takeaway, the portfolio reflection in the README), and confirm the account of the collaboration in Question 4 matches what happened.
