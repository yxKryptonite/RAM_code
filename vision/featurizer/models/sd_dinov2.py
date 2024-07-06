import torch
from vision.featurizer.models.dinov2 import DINOv2Featurizer
from vision.featurizer.models.dift_sd import SDFeaturizer

class SD_DINOv2Featurizer:
    def __init__(self, 
                 sd_id='stabilityai/stable-diffusion-2-1', 
                 dino_id='dinov2_vitb14',
                 null_prompt=''):
        self.sd_model = SDFeaturizer(sd_id, null_prompt)
        self.dinov2_model = DINOv2Featurizer(dino_id)
        
    def forward(self,
                img_tensor,
                block_index,
                prompt,
                ensemble_size,
                RESOLUTION=64):
        sd_out = self.sd_model.forward(img_tensor, block_index=block_index, prompt=prompt, ensemble_size=ensemble_size) # 1,c,h1,w1
        dino_out = self.dinov2_model.forward(img_tensor, block_index, prompt, ensemble_size) # 1,c,h2,w2
        # normalize c
        sd_out = sd_out / sd_out.norm(dim=1, keepdim=True)
        dino_out = dino_out / dino_out.norm(dim=1, keepdim=True)
        # resize and concat
        sd_out = torch.nn.functional.interpolate(sd_out, size=(RESOLUTION, RESOLUTION), mode='bilinear', align_corners=False)
        dino_out = torch.nn.functional.interpolate(dino_out, size=(RESOLUTION, RESOLUTION), mode='bilinear', align_corners=False)
        out = torch.cat((sd_out, dino_out), dim=1)
        return out