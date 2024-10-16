from PIL import Image
import random
from .base_model import BaseModel

class NoiseModel(BaseModel):
    def process_images(self, image_path):
        image = Image.open(image_path)
        # Adding simple noise by modifying pixel values
        noisy_image = image.point(lambda p: p + random.randint(-50, 50))
        return noisy_image