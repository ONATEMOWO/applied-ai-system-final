# assets/

Reserved for static project assets (for example an optional exported PNG of
[diagrams/architecture.mmd](../diagrams/architecture.mmd)).

This project's evidence is deliberately **text-based and reproducible**, so there are no
screenshots here. Generated evidence lives in:

* [logs/](logs/): captured output of the evaluation harness and other commands
* [../traces/](../traces/): JSON runtime traces of the application agent

## Regenerating the logs

The saved logs are the real output of these commands at the final commit
(`tests/test_docs.py` fails if they go stale, except `pytest.txt`, whose timing varies; that run
deselects the one test that reads `pytest.txt` itself, so the log reads "N-1 passed, 1 deselected"):

```bash
python3 -m src.main --width 100        > assets/logs/src_main.txt
python3 -m src.assistant --demo        > assets/logs/assistant_demo.txt
python3 -m src.evaluate                > assets/logs/evaluate.txt
python3 -m pytest -q -k "not test_the_test_count_quoted_in_the_docs_is_the_real_count" > assets/logs/pytest.txt
```
