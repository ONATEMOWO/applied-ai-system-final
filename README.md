# VibeFinder AI: Intelligent Music Discovery Assistant

> **Status: work in progress.** This README is replaced with the full documentation when the implementation is finished. The design is in [docs/design.md](docs/design.md).

VibeFinder AI extends the Module 3 **Music Recommender Simulation** with a natural-language assistant that interprets a listening request, retrieves project knowledge, plans a recommendation, calls the original recommendation engine, validates the result, and explains it.

## Original project

The Module 3 project is a content-based music recommender over 30 fictional songs with 15 attributes, with weighted scoring, accurate per-song explanations, three ranking modes, an optional variety rerank, and a CLI. Its documentation is preserved unchanged in [docs/module3/](docs/module3/README.md), and the original repository is
<https://github.com/ONATEMOWO/ai110-module3show-musicrecommendersimulation-starter>.

The original entry point still works:

```
python3 -m pytest
python3 -m src.main
```
