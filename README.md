# slm

The plan according to [Karpathys' post]:
1. generate small dataset using synthetic dataset technique described in [HumanEval paper]
2. Baselines on HumanEval:
    1. Baseline model - gpt2
    2. Human baseline
    3. Input-independent model baseline (train with all inputs set to 0, but real targets)
    4. gpt2 overfitting one batch
3. Try scaling baseline model, verify this helps with test


## Holdout Dataset for testing small models Perplexity

```py
eval_texts = [
    "The cat sat on the mat.", # human-like text
    "Once upon a time, there was a little girl who lived in a forest.", # Tiny stories like text
    "The sun rises in the east and sets in the west.", # World knowledge
    "One plus one is equal to two.", # Math
    "If it is raining outside, you should take an umbrella.", # Logic
]
```

To evaluate model run `uv run download.py` and pass huggingface repo with model with compatible architecture.
For example: `mikeoxmaul/zmeeust-w`


## Experiments (in order of creation)
Only good to go experiments, others I'm still considering if are good enough for learning/evaluation purposes.

### gpt\_neo\_wikitext2
Same dataset, gpt neo archtecture as seen in [Tiny Stories paper]

Perplexity on holdout dataset: 1061.24

### gpt\_neo\_tiny\_stories
Tiny stories dataset, gpt neo, hyperparameters all as seen in [Tiny Stories paper]

Perplexity on holdout dataset: 25.39

### gpt\_neo\_babylm
Babylm dataset, gpt neo, hyperparameters all as seen in [Tiny Stories paper]. More info about [Babylm challenge] and [Babylm dataset]

Perplexity on holdout dataset: 35.12

## Dev

`uv sync` to install deps.

`uv run main.py --max_steps=100 --num_samples=10` launches latest edited experiment from `experiments/` folder
with 100 max steps on first 10 samples from dataset.
By default num\_samples is picked according to [Chinchilla scaling laws]  ~ 20 tokens per 1 trainable parameter.
Run script is based around the idea that just `uv run main.py` would run full training of the latest (best) model without extra thinking.

`uvx tensorboard --logdir lightning_logs/` to watch at the curves.


See [onnx notebook](https://colab.research.google.com/drive/1fwTNiZS1TaUsm4v5h_B31_BFuG-ap2gF?usp=sharing) for optimizing model to run in browser.

## Run

use `uv run chat.py` to evaluate latest model. It is not chat, just prompting (text completion).


[Karpathys' post]: https://karpathy.github.io/2019/04/25/recipe/
[HumanEval paper]: https://arxiv.org/abs/2107.03374
[Tiny Stories paper]: https://arxiv.org/abs/2305.07759
[WikiText2 dataset]: https://github.com/pytorch/examples/tree/main/word_language_model/data/wikitext-2
[Chinchilla scaling laws]: https://arxiv.org/abs/2203.15556
[Babylm challenge]: https://babylm.github.io/
[Babylm dataset]: https://huggingface.co/datasets/nilq/babylm-10M
