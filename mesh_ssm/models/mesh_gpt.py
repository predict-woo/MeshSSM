import lightning as L
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class MeshGPT(L.LightningModule):
    def __init__(
        self,
        config: GPT2Config,
        lr=1e-4,
        weight_decay=0.01,
        max_epochs=100,
    ):
        super(MeshGPT, self).__init__()
        self.save_hyperparameters()

        self.config = config
        self.model = GPT2LMHeadModel(self.config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        result = self(input_ids, attention_mask, labels)
        loss = result.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
