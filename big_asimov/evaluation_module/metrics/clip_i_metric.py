import torch
import clip
from .base_metric import BaseMetric

class CLIPIMetric(BaseMetric):
    def __init__(self, device):
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def evaluate(self, input_image, output_image):
        input_image = self.preprocess(input_image).unsqueeze(0).to(self.device)
        output_image = self.preprocess(output_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            input_features = self.clip_model.encode_image(input_image)
            output_features = self.clip_model.encode_image(output_image)

        score = torch.nn.functional.cosine_similarity(input_features, output_features).item()
        return score