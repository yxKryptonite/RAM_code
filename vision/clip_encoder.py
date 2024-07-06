import torch
# import clip
import open_clip
from PIL import Image

# model, preprocess = clip.load("ViT-B/32", device='cuda')
# tokenizer = clip.tokenize

class ClipModel:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda') -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)


    def get_vision_feature(self, img: Image.Image):
        with torch.no_grad():
            image = self.preprocess(img).unsqueeze(0).cuda()
            img_ft = self.model.encode_image(image)
            img_ft /= img_ft.norm(dim=-1, keepdim=True)
            return img_ft


    def get_text_feature(self, text: str):
        with torch.no_grad():
            text = self.tokenizer([text]).cuda()
            text_ft = self.model.encode_text(text)
            text_ft /= text_ft.norm(dim=-1, keepdim=True)
            return text_ft
        
    def compute_similarity(self, feat1, feat2):
        return (feat1 @ feat2.T).cpu().numpy()