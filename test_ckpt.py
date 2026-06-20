import torch
from tcpformer_model import MemoryInducedTransformer

ckpt = torch.load("TCPFormer_h36m_81_405.pth.tr", map_location='cpu', weights_only=False)
state_dict = ckpt['model']

with open("ckpt_keys.txt", "w") as f:
    for k in state_dict.keys():
        f.write(f"{k} {state_dict[k].shape}\n")
