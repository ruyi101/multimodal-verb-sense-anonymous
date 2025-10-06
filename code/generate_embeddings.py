import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from transformers import MllamaProcessor, MllamaForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import json
from tqdm import tqdm


from huggingface_hub import login
login(token='')


# load model
device='cuda' if torch.cuda.is_available() else 'cpu'
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = MllamaProcessor.from_pretrained(model_name)
model = MllamaForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(device)


def embed_image_text(image_path, text_prompt=None):
        """
        Compute embeddings for an image and optionally concatenate with text embeddings (for CLIP multimodal).

        Args:
            image_path (str): Path to the image file.
            text_prompt (str, optional): Text prompt for multimodal embeddings.

        Returns:
            np.ndarray: Combined embedding vector for the image and text.
        """
        image = Image.open(image_path).convert("RGB")


        if text_prompt is None:
            text_prompt = '<|image|>'
        else:
            text_prompt = f'<|image|> {text_prompt}'

        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states

        return hidden_states[-1][:, -1, :].cpu().numpy().squeeze()

########################### change here for paths ######################################
image_folder = 'images_resized'
outfile = 'Results/llama_504_image_verb_embeddings.json'
data = json.load(open('Data/llama_504_verbs_rows.json'))


# Extract the folder path from the outfile
folder = os.path.dirname(outfile)
# Create the folder if it doesn't exist
if folder and not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Folder '{folder}' is ready.")




paths = data['image']
paths = [f'{image_folder}/{im}' for im in paths]
prompts = data['verbs']





embeddings = []
for i, image_path in enumerate(tqdm(paths, desc="Embedding images")):
    embedding = embed_image_text(image_path, text_prompt=prompts[i] if prompts else None)
    embeddings.append(embedding.tolist())

data['embedding'] = embeddings


with open(outfile, 'w') as f:
    json.dump(data, f)