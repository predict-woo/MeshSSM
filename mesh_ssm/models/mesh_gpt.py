import pytorch_lightning as pl
from transformers import GPT2Config, GPT2LMHeadModel
import torch


class TransformerModel(pl.LightningModule):
    def __init__(self, n_positions=1024, n_ctx=1024, n_embd=192, n_layer=12, n_head=12):
        super().__init__()
        config = GPT2Config(
            vocab_size=1,  # Not using a vocab
            n_positions=n_positions,  # Adjust as necessary
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.model = GPT2LMHeadModel(config)
        self.model.transformer.wte = torch.nn.Identity()

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5)
