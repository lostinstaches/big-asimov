import os
from big_asimov.models.noise_model import NoiseModel
from big_asimov.models.pixelate_model import PixelateModel

class ModelInfrastructure:
    def __init__(self, models, input_dir, output_dir):
        self.models = models
        self.input_dir = input_dir
        self.output_dir = output_dir

    def start(self):
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for model in self.models:
            model_output_dir = os.path.join(self.output_dir, model.model_id)
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)

            for image_name in os.listdir(self.input_dir):
                image_path = os.path.join(self.input_dir, image_name)
                if os.path.isfile(image_path):
                    processed_image = model.process_images(image_path)
                    output_path = os.path.join(model_output_dir, image_name)
                    processed_image.save(output_path)