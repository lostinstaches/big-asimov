class BaseModel:
    def __init__(self, model_id):
        self.model_id = model_id

    def process_images(self, image_path):
        raise NotImplementedError("Subclasses must implement process_images method")