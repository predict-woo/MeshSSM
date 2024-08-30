from mesh_ssm.models.mesh_gpt import MeshGPT
import torch
from tools import Tools
import numpy as np
from project.render import render_mesh

from data import EncodedChairDataset
from torch.utils.data import DataLoader


device = "cuda:3"

dataset = EncodedChairDataset("encoded_chair_dataset.npy")

data = dataset[1]

inp = data["input_ids"]
print(inp)
inp = inp[:24]

inp = np.expand_dims(inp, 0)


tools = Tools(device=device)
tools.init_autoencoder(
    "checkpoints/chair-final-bs[128]-lr[0.001]-ks[3]/epoch=987-val_loss=0.0000.ckpt"
)

path = "checkpoints/chair_gpt_global_codebook-lr[0.001]-wd[0.01]-bs[12]/epoch=186-val_loss=0.59.ckpt"

with torch.cuda.device(device):
    model = MeshGPT.load_from_checkpoint(
        path,
        map_location=device,
    )

    model.eval()

    inp = torch.tensor(inp, device=device)

    print("input", inp)
    output = model.generate(
        inp,
        attention_mask=None,
        max_length=800 * 2 * 3 + 2,
        temperature=0.1,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=2,
        repetition_penalty=1,
    )

    print("output", output.sequences)

    tokens = tools.sequence_to_tokens(output.sequences)

    print("tokens", tokens)

    mesh = tools.decode_mesh(tokens)

    verts = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()

    image = render_mesh(verts, faces)
    image.save("res2.png")
