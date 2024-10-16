from PIL import Image
import random
from .base_model import BaseModel

class HardNoiseModel(BaseModel):
    def process_images(self, image_path):
        image = Image.open(image_path)
        # Adding hard noise by modifying pixel values with a higher intensity
        noisy_image = image.point(lambda p: p + random.randint(-100, 100))
        return noisy_image