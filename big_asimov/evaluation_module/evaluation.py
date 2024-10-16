import os
from PIL import Image
import torch
from .metrics.clip_i_metric import CLIPIMetric
from .metrics.dino_metric import DINOMetric
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from pandas.plotting import table
import torch
import clip
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch.hub
from tqdm import tqdm
import time

class EvaluationSystem:
    def __init__(self, models, input_dir, output_dir):
        self.models = models
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = [
            CLIPIMetric(device=self.device),
            DINOMetric(device=self.device)
        ]

    def start_evaluation(self):
        evaluation_results = []

        for model in self.models:
            scores = self.evaluate_model(model)
            evaluation_results.append({
                'model_id': model.model_id,
                **scores
            })

        self.visualize_results(evaluation_results)

    def evaluate_model(self, model):
        model_output_dir = os.path.join(self.output_dir, model.model_id)
        scores = {metric.__class__.__name__: 0 for metric in self.metrics}
        image_count = 0

        for image_name in tqdm(os.listdir(model_output_dir), desc=f"Evaluation for model {model.model_id}"):
            input_image_path = os.path.join(self.input_dir, image_name)
            output_image_path = os.path.join(model_output_dir, image_name)

            if os.path.isfile(input_image_path) and os.path.isfile(output_image_path):
                input_image = Image.open(input_image_path).convert('RGB')
                output_image = Image.open(output_image_path).convert('RGB')

                for metric in self.metrics:
                    score = metric.evaluate(input_image, output_image)
                    scores[metric.__class__.__name__] += score

                image_count += 1

        if image_count > 0:
            for key in scores.keys():
                scores[key] /= image_count

        return scores

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

    def dino_evaluation(self, input_image_path, output_image_path):
        # Load and preprocess images
        input_image = Image.open(input_image_path).convert('RGB')
        output_image = Image.open(output_image_path).convert('RGB')

        input_image = self.dino_feature_extractor(input_image).unsqueeze(0).to(self.device)
        output_image = self.dino_feature_extractor(output_image).unsqueeze(0).to(self.device)

        # Get image embeddings using DINO
        with torch.no_grad():
            with torch.no_grad():
                input_features = self.dino_model(input_image)
            output_features = self.dino_model(output_image)

        # Compute cosine similarity
        dino_score = torch.nn.functional.cosine_similarity(input_features, output_features, dim=-1).mean().item()
        return dino_score

    def visualize_results(self, evaluation_results):
        df = pd.DataFrame(evaluation_results)
        self.print_results(df)
        self.plot_results(df, [metric.__class__.__name__ for metric in self.metrics])

    def print_results(self, df):
        # Print the evaluation results as a clean table
        print(df)

    def plot_results(self, df, metrics):
        # Highlight the winning model in subtle green using a separate column
        for metric in metrics:
            max_score = df[metric].max()
            df[f'highlight_{metric}'] = df[metric].apply(
                lambda x: 'background-color: lightgreen' if x == max_score else '')

        # Plot the evaluation results as a table and save as a PNG file
        fig, ax = plt.subplots(figsize=(12, 8))  # Make the figure larger
        ax.axis('off')
        ax.axis('tight')
        tbl = table(ax, df.drop(columns=[col for col in df.columns if 'highlight' in col]).reset_index(drop=True),
                    loc='center', cellLoc='center', colWidths=[0.3] * len(df.columns))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.5, 1.5)  # Scale the table for better readability
        # Apply highlight colors
        for key, cell in tbl.get_celld().items():
            for metric in metrics:
                if key[0] > 0 and key[1] == metrics.index(metric) + 1 and df.iloc[key[0] - 1][
                    f'highlight_{metric}'] == 'background-color: lightgreen':
                    cell.set_facecolor('lightgreen')

        if not os.path.exists('results'):
            os.makedirs('results')
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        plt.savefig(os.path.join('results', f'evaluation_results_{timestamp}.png'), bbox_inches='tight', dpi=300)
        plt.show()