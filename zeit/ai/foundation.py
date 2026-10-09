import torch
import torch.nn as nn
from transformers import AutoModel

class GeoFoundationViT(nn.Module):
    """
    Wrapper for NASA/IBM Prithvi-100M Foundation Model via HuggingFace.
    """
    def __init__(self, model_id='ibm-nasa-geospatial/Prithvi-100M', num_classes=2):
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.has_hf = True
        except Exception:
            self.backbone = nn.Identity()
            self.fallback_conv = nn.Conv3d(6, 768, kernel_size=(1, 16, 16), stride=(1, 16, 16))
            self.has_hf = False
            
        self.classifier = nn.Conv2d(768, num_classes, kernel_size=1)
        
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if not self.has_hf:
            feats = self.fallback_conv(x).mean(dim=2)
            # back to the input's size (not x16: a side that is not a multiple of 16 would shrink)
            feats = nn.functional.interpolate(feats, size=x.shape[-2:], mode='bilinear')
            return self.classifier(feats)
        
        outputs = self.backbone(pixel_values=x)
        last_hidden = outputs.last_hidden_state
        B, Seq, Dim = last_hidden.shape
        H = W = int(Seq ** 0.5)
        feats = last_hidden.permute(0, 2, 1).view(B, Dim, H, W)
        feats = nn.functional.interpolate(feats, size=x.shape[-2:], mode='bilinear')
        return self.classifier(feats)
