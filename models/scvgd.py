import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class SCVGD(nn.Module):
    def __init__(
        self,
        vit_branch,
        cnn_branch,
        fusion_module,
        num_classes,
        embed_dim=256
    ):
        super().__init__()

        self.vit_branch = vit_branch
        self.cnn_branch = cnn_branch
        self.fusion = fusion_module

        self.classifier = Classifier(embed_dim, num_classes)

    def forward(self, x, return_aux=False):
        """
        x: input image [B, 3, H, W]
        """

        # Transformer branch
        vit_out = self.vit_branch(x)

        # CNN branch
        cnn_out = self.cnn_branch(x)

        # Fusion
        embedding, part_visibility = self.fusion(
            vit_out=vit_out,
            cnn_out=cnn_out,
            cnn_feat_map=None   # can be added later
        )

        # Classification logits
        logits = self.classifier(embedding)

        if return_aux:
            return {
                "embedding": embedding,
                "logits": logits,
                "part_visibility": part_visibility,
                "patch_visibility": vit_out["visibility"],
                "semantic_maps": cnn_out["semantic_maps"]
            }

        return embedding, logits
