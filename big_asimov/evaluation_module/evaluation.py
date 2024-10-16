import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import torch
import clip
from PIL import Image

class EvaluationSystem:
    def __init__(self, models, input_dir, output_dir):
        self.models = models
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def start_evaluation(self):
        evaluation_results = []

        for model in self.models:
            scores = self.evaluate_model(model)
            evaluation_results.append({
                'model_id': model.model_id,
                'average_clip_i_score': float(scores['average_clip_i_score'])
            })

        self.visualize_results(evaluation_results)

    def evaluate_model(self, model):
        model_output_dir = os.path.join(self.output_dir, model.model_id)
        total_clip_i_score = 0
        image_count = 0

        for image_name in os.listdir(model_output_dir):
            input_image_path = os.path.join(self.input_dir, image_name)
            output_image_path = os.path.join(model_output_dir, image_name)

            if os.path.isfile(input_image_path) and os.path.isfile(output_image_path):
                clip_i_score = self.clip_i_evaluation(input_image_path, output_image_path)
                total_clip_i_score += clip_i_score
                image_count += 1

        if image_count > 0:
            average_clip_i_score = total_clip_i_score / image_count
        else:
            average_clip_i_score = 0

        return {
            'average_clip_i_score': average_clip_i_score
        }

    def clip_i_evaluation(self, input_image_path, output_image_path):
        # Load and preprocess images
        input_image = Image.open(input_image_path)
        output_image = Image.open(output_image_path)
        input_image = self.clip_preprocess(input_image).unsqueeze(0).to(self.device)
        output_image = self.clip_preprocess(output_image).unsqueeze(0).to(self.device)

        # Get image embeddings
        with torch.no_grad():
            input_features = self.clip_model.encode_image(input_image)
            output_features = self.clip_model.encode_image(output_image)

        # Compute cosine similarity
        clip_i_score = torch.nn.functional.cosine_similarity(input_features, output_features).item()
        return clip_i_score

    def visualize_results(self, evaluation_results):
        df = pd.DataFrame(evaluation_results)
        self.print_results(df)
        self.plot_results(df)

    def print_results(self, df):
        # Print the evaluation results as a clean table
        print(df)

    def plot_results(self, df):
        # Highlight the winning model in subtle green using a separate column
        max_score = df['average_clip_i_score'].max()
        df['highlight'] = df['average_clip_i_score'].apply(
            lambda x: 'background-color: lightgreen' if x == max_score else '')

        # Plot the evaluation results as a table and save as a PNG file
        fig, ax = plt.subplots(figsize=(12, 8))  # Make the figure larger
        ax.axis('off')
        ax.axis('tight')
        tbl = table(ax, df.drop(columns=['highlight']).reset_index(drop=True), loc='center', cellLoc='center',
                    colWidths=[0.3] * len(df.columns))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.5, 1.5)  # Scale the table for better readability
        # Apply highlight colors
        for key, cell in tbl.get_celld().items():
            if key[0] > 0 and key[1] == 1 and df.iloc[key[0] - 1]['highlight'] == 'background-color: lightgreen':
                cell.set_facecolor('lightgreen')
        plt.savefig('evaluation_results_table.png', bbox_inches='tight', dpi=300)
        plt.show()