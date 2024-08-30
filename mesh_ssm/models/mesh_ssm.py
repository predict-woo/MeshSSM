import lightning as L
import numpy as np
import os
import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.optim import AdamW


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
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.model = MambaLMHeadModel(self.config)

        self.results_dir = f"results/mesh_ssm_lr[{self.lr}]"

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

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
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        # loss_fct = nn.CrossEntropyLoss()

        # Shift logits and labels to match shape: (batch_size * seq_len, vocab_size)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        result = self(input_ids, labels)
        loss = self.compute_loss(result.logits, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        result = self(input_ids, labels)
        loss = self.compute_loss(result.logits, labels)

        self.log("val_loss", loss)
        return loss

    def on_validation_end(self) -> None:
        # if on first gpu
        inp = torch.tensor([[1]], device=self.device)
        output = self.generate(
            inp,
            max_length=800 * 2 * 3 + 2,
            temperature=0.7,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        print(output.sequences)

        if self.trainer.is_global_zero:
            np.save(
                os.path.join(self.results_dir, f"epoch[{self.current_epoch}].npy"),
                output.sequences.detach().cpu().numpy(),
            )

    def generate(
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        repetition_penalty=1,
        **kwargs,
    ):
        return self.model.generate(
            input_ids,
            max_length,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer
