import torch
import lightning as L
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DDPStrategy


model = BoringModel()
trainer = L.Trainer(max_epochs=10, devices=[7, 8])

trainer.fit(model)
print("Done")
