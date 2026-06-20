import torch
from tcpformer_model import MemoryInducedTransformer

ckpt = torch.load("TCPFormer_h36m_81_405.pth.tr", map_location='cpu', weights_only=False)
state_dict = ckpt['model']

# Create model
model = MemoryInducedTransformer(
    n_layers=14,
    dim_in=3,
    dim_feat=128,
    dim_rep=512,
    dim_out=3,
    n_frames=81
)

# Strip module. prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

# Dummy input
x = torch.zeros(1, 81, 17, 3)
y = model(x)
print("Output shape:", y.shape)
