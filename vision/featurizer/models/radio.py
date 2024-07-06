import torch
from einops import rearrange

class RADIOFeaturizer:
    def __init__(self, radio_id='radio_v2'):
        self.radio_id = radio_id
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=radio_id, progress=True)
        self.model.cuda().eval()

    @torch.no_grad()
    def forward(self, img_tensor, block_index, prompt=None, ensemble_size=None):
        patch_size = 16
        summary, spatial_features = self.model(img_tensor) # RADIO: spatial features have shape (B, T, D) with T being the flattened spatial tokens, and D being the channels for spatial features.
        spatial_features = rearrange(spatial_features, 'b (h w) d -> b d h w', h=img_tensor.shape[-2] // patch_size, w=img_tensor.shape[-1] // patch_size) # (B, D, H, W)
        return spatial_features
