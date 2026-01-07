import timm
from models.vit_branch import ViTBranch
from models.vit_pytorch import vit_base_patch16_224_TransReID
from models.cnn_branch import CNNBranch
from models.fusion import VisibilityGuidedFusion
from models.scvgd import SCVGD
import torch

# Build ViT backbone
vit = vit_base_patch16_224_TransReID(
    "deit_small_patch16_224",
    pretrained=True,
    num_classes=0
)

vit_branch = ViTBranch(vit, embed_dim=384)
cnn_branch = CNNBranch(num_parts=5, embed_dim=256)
fusion = VisibilityGuidedFusion(
    embed_dim=256,
    num_parts=5,
    num_patches=128
)

model = SCVGD(
    vit_branch=vit_branch,
    cnn_branch=cnn_branch,
    fusion_module=fusion,
    num_classes=751
)

x = torch.randn(2, 3, 256, 128)
emb, logits = model(x)

print(emb.shape)     # [2, 256]
print(logits.shape)  # [2, 751]
