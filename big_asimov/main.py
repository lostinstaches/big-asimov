from infrastructure_module.infrastructure import ModelInfrastructure
from models.noise_model import NoiseModel
from models.pixelate_model import PixelateModel
from evaluation_module.evaluation import EvaluationSystem

if __name__ == "__main__":
    # Set up models and directories
    models = [NoiseModel("model_noise"), PixelateModel("model_pixelate")]
    input_dir = "data"
    output_dir = "processed_images"

    # Start model infrastructure to process images
    model_infrastructure = ModelInfrastructure(models, input_dir, output_dir)
    model_infrastructure.start()

    print("Image processing complete. Check the processed_images folder for results.")

    # Start evaluation
    evaluation_system = EvaluationSystem(models, input_dir, output_dir)
    evaluation_system.start_evaluation()