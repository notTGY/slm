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
Only good to go experiments, others I'm still considering if are good enough for learning/evaluation purposes.

### simple\_transformer\_wikitext2
Almost entirely copied from documentation of lightning. Dataset is [WikiText2 dataset]

### gpt\_neo\_wikitext2
Same dataset, gpt neo archtecture as seen in [Tiny Stories paper]

### gpt\_neo\_tiny\_stories
Tiny stories dataset, gpt neo, hyperparameters all as seen in [Tiny Stories paper]

## Dev

`uv sync` to install deps.

`uv run main.py --max_steps=100 --num_samples=10` launches latest edited experiment from `experiments/` folder
with 100 max steps on first 10 samples from dataset.
By default num\_samples is picked according to [Chinchilla scaling laws]  ~ 20 tokens per 1 trainable parameter.
Run script is based around the idea that just `uv run main.py` would run full training of the latest (best) model without extra thinking.

`uvx tensorboard --logdir lightning_logs/` to watch at the curves.

## Run

use `uv run chat.py` to evaluate latest model. It is not chat, just prompting (text completion).


[Karpathys' post]: https://karpathy.github.io/2019/04/25/recipe/
[HumanEval paper]: https://arxiv.org/abs/2107.03374
[Tiny Stories paper]: https://arxiv.org/abs/2305.07759
[WikiText2 dataset]: https://github.com/pytorch/examples/tree/main/word_language_model/data/wikitext-2
[Chinchilla scaling laws]: https://arxiv.org/abs/2203.15556
