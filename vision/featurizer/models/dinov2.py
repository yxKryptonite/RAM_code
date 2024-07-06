import torch
import einops as E

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

class DINOv2Featurizer:
    def __init__(self, dino_id='dinov2_vitb14'):
        self.model = torch.hub.load('facebookresearch/dinov2', dino_id).eval().cuda()

    @torch.no_grad()
    def forward(self, img_tensor, block_index, prompt=None, ensemble_size=None):
        # img_tensor: 1, 3, H, W
        h = img_tensor.shape[2] // 14
        w = img_tensor.shape[3] // 14
        x = self.model.prepare_tokens_with_masks(img_tensor, None)
        
        num_layers = len(self.model.blocks) # 12 blocks
        n = num_layers - block_index
        
        embeds = []
        for i, blk in enumerate(self.model.blocks): 
            x = blk(x)
            embeds.append(x)
            
        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output("dense", spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[n] #[1, 768, 32, 32]
        
    