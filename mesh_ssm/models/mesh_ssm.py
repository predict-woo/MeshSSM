import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class MeshSSM(L.LightningModule):
    def __init__(
        self,
        config: MambaConfig,
        lr=1e-4,
        weight_decay=0.01,
        max_epochs=100,
    ):
        super(MeshSSM, self).__init__()
        self.save_hyperparameters()

        self.config = config
        self.model = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def setup(self, stage=None):
        self.model = MambaLMHeadModel(self.config, device=self.device, dtype=self.dtype)

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        return self.model(
            input_ids, position_ids, inference_params, num_last_tokens, **mixer_kwargs
        )

    def compute_loss(self, logits, labels):
        # Assuming the model is used for language modeling, which often uses CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss()
        # Shift logits and labels to match shape: (batch_size * seq_len, vocab_size)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        result = self(input_ids, labels)
        loss = self.compute_loss(result.logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
