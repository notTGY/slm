1. Project Gutenberg provides super-simple api to download txt books. https://lightning.ai/docs/examples/train-models-low-code
```
curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt
```
For exapmle compare how different books impact perplexity/GLUE.

2. Test different optimizers and schedulers:
   - AdamW optimizer (currently using SGD)
   - Learning rate warmup + cosine decay schedule
