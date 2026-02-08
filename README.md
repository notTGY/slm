# slm

The plan according to [Karpathys' post]:
1. generate small dataset using synthetic dataset technique described in [HumanEval paper]
2. Baselines on HumanEval:
    1. Baseline model - gpt2
    2. Human baseline
    3. Input-independent model baseline (train with all inputs set to 0, but real targets)
    4. gpt2 overfitting one batch
3. Try scaling baseline model, verify this helps with test


## Experiments (in order of creation)

### simple\_transformer\_wikitext2
Almost entirely copied from documentation of lightning.

### gpt\_neo\_wikitext2
Same dataset, gpt neo archtecture as seen in [Tiny Stories paper]

### gpt\_neo\_tiny\_stories
Tiny stories dataset, gpt neo, hyperparameters all as seen in [Tiny Stories paper]

## Dev

`uv sync` to install deps.

`uv run main.py` launches latest edited experiment from `experiments/` folder.

`uvx tensorboard --logdir lightning_logs/` to watch at the curves.


[Karpathys' post]: https://karpathy.github.io/2019/04/25/recipe/
[HumanEval paper]: https://arxiv.org/abs/2107.03374
[Tiny Stories paper]: https://arxiv.org/abs/2305.07759
