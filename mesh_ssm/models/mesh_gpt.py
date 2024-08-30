import lightning as L
import numpy as np
import os
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW


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
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.model = GPT2LMHeadModel(self.config)

        self.results_dir = f"results/mesh_gpt_lr[{self.lr}]"

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
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
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        result = self(input_ids, attention_mask, labels)
        loss = self.compute_loss(result.logits, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        result = self(input_ids, attention_mask, labels)
        loss = self.compute_loss(result.logits, labels)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer

        # input_ids = (input_ids,)
        # attention_mask = (attn_mask,)
        # max_length = (max_length,)
        # return_dict_in_generate = (True,)
        # pad_token_id = (tokenizer.eos_token_id,)
        # do_sample = (True,)
        # temperature = (args.temperature,)
        # top_k = (args.topk,)
        # top_p = (args.topp,)
        # repetition_penalty = (args.repetition_penalty,)

    def generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        return_dict_in_generate=True,
        pad_token_id=0,
        do_sample=True,
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        **kwargs,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            return_dict_in_generate=return_dict_in_generate,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            **kwargs,
        )
