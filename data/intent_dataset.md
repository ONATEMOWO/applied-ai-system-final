# Intent dataset (version v1)

Synthetic, hand-written requests used to train and evaluate the small intent classifier in
[src/intent.py](../src/intent.py). **None of this is real user data.** Every sentence was
written by the coding agent for this project and then reviewed only by automated checks; it
does not represent how real listeners phrase requests.

| File | Purpose | Rows | Labels |
|---|---|---|---|
| `intent_train_v1.csv` | trains the Naive Bayes model | 144 (24 per label) | study, workout, party, relax, discovery, unsupported |
| `intent_dev_v1.csv` | development only: used to choose the model's smoothing and abstention threshold | 34 | the six labels above plus `uncertain` |
| `intent_heldout_v1.csv` | final comparison of the keyword baseline against the specialized model | 66 | the six labels above plus `uncertain` |

* `uncertain` is the *expected abstention* label: a request with no usable listening goal
  ("music please"). It is never a training class; the model abstains when its best relative
  score is low or when none of the request's words were seen in training.
* `unsupported` covers out-of-domain requests (weather, translation, streaming, downloads).
* The held-out file was written before any classifier code existed and was not used to
  choose any setting. `tests/test_intent.py` fails if a sentence appears in more than one
  file (after lowercasing and removing punctuation).
* The sets are small. Accuracy on 66 examples has wide uncertainty, and the vocabulary of the
  three files overlaps heavily because one author wrote them all.
