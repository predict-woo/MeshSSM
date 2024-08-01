import torch
import lightning as L
from lightning.pytorch.demos.boring_classes import BoringModel

ngpus = 1

model = BoringModel()
trainer = L.Trainer(max_epochs=10, devices=ngpus)

trainer.fit(model)
print("Done")
