# slm

The plan according to [karpathy post](https://karpathy.github.io/2019/04/25/recipe/):
1. generate small dataset using synthetic dataset technique described in [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
2. Baselines on HumanEval:
    1. Baseline model - gpt2
    2. Human baseline
    3. Input-independent model baseline (train with all inputs set to 0, but real targets)
    4. gpt2 overfitting one batch
3. Try scaling baseline model, verify this helps with test


## History

### simple\_transformer\_wikitext2
Almost entirely copied from documentation of lightning.

`gpt_neo_wikitext2`

`gpt_neo_tiny_stories`

## Dev

`uv sync` to install deps.

`uv run main.py` launches latest edited experiment from `experiments/` folder.
