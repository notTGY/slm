import pytorch_lightning as pl

from models import LightningTransformer
from datamodules import WikiText2

def main(max_steps=-1):
    dataset = WikiText2()
    vocab_size = dataset.vocab_size

    model = LightningTransformer(vocab_size=vocab_size)

    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=max_steps,
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()
