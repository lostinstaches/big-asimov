from PIL import Image
from .base_model import BaseModel

class HardPixelateModel(BaseModel):
    def process_images(self, image_path):
        image = Image.open(image_path)
        # Pixelate the image by resizing it down and then back up
        small_image = image.resize((image.width // 35, image.height // 35), resample=Image.NEAREST)
        pixelated_image = small_image.resize(image.size, Image.NEAREST)
        return pixelated_image