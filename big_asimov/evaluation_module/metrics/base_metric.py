class BaseMetric:
    def evaluate(self, input_image, output_image):
        """
        Evaluates the similarity or quality of two images.
        Should be implemented by subclasses.

        Args:
            input_image: The original input image.
            output_image: The generated output image.

        Returns:
            A numerical score representing the evaluation.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method")