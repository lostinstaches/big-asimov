from infrastructure_module.infrastructure import ModelInfrastructure
from models.noise_model import NoiseModel
from models.hard_noise_model import HardNoiseModel
from models.pixelate_model import PixelateModel
from models.hard_pixelate_model import HardPixelateModel
from evaluation_module.evaluation import EvaluationSystem

if __name__ == "__main__":
    # Set up models and directories
    models = [
        NoiseModel("model_noise"),
        PixelateModel("model_pixelate"),
        HardNoiseModel("model_hard_noise"),
        HardPixelateModel("model_hard_pixelate"),
    ]
    input_dir = "data"
    output_dir = "processed_images"

    # Start model infrastructure to process images
    model_infrastructure = ModelInfrastructure(models, input_dir, output_dir)
    model_infrastructure.start()

    print("Image processing complete. Check the processed_images folder for results.")

    evaluation_system = EvaluationSystem(models, input_dir, output_dir)
    evaluation_system.start_evaluation()