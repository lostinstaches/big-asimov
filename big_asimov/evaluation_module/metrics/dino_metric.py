import torch
import torchvision.transforms as T
from .base_metric import BaseMetric

class DINOMetric(BaseMetric):
    def __init__(self, device):
        self.device = device
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(self.device)
        self.feature_extractor = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def evaluate(self, input_image, output_image):
        input_image = self.feature_extractor(input_image).unsqueeze(0).to(self.device)
        output_image = self.feature_extractor(output_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            input_features = self.dino_model(input_image)
            output_features = self.dino_model(output_image)

        score = torch.nn.functional.cosine_similarity(input_features, output_features, dim=-1).mean().item()
        return score