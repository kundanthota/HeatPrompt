
import os
import json
import cv2
import numpy as np
from openai import OpenAI
import yaml
from dotenv import load_dotenv
import requests
import json
import base64
import argparse
import warnings

warnings.filterwarnings("ignore")

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
parser = argparse.ArgumentParser(description="Fetch WAD features by BBOX.")
parser.add_argument('--model', required=True, help='Model to use for inference')

args = parser.parse_args()
model = args.model

if model == 'gpt-4o':
    inference_model = "openai/gpt-4o"
elif model == 'qwen2.5':
    inference_model = "qwen/qwen2.5-vl-72b-instruct"
else:
    raise ValueError("Unsupported model specified. Use 'gpt-4o' or 'qwen2.5'.")


# Load system prompt from config.yml
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
system_prompt = config["system"]["prompt"]
temperature = config["system"]["temperature"]

# Initialize OpenAI client with key from .env
client = OpenAI(api_key=OPENAI_API_KEY)

annotations_file = f"data/annotations/annotations_{model}.json"

# Load existing annotations if they exist
if os.path.exists(annotations_file):
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
else:
    annotations = {}

def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

# Load image keys from data/images/
images_dir = "data/images"
image_files = [f for f in os.listdir(images_dir) if f.startswith("image_") and f.endswith(".png")]
data_keys = [f.split("_")[1].split(".")[0] for f in image_files]



def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

for key in data_keys:
    if key in annotations:
        continue  # Skip if already processed

    try:
        # Read the main image and mask
        image_path = os.path.join(images_dir, f"image_{key}.png")
        mask_path = os.path.join(images_dir, f"mask_{key}.png")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)  # nearest keeps binary edges sharp

        # Match mask channels to image
        if len(image.shape) == 3 and mask.ndim == 2:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Apply mask
        result = (image * mask).astype(np.uint8)
        masked_image_path = f"masked_image.png"
        cv2.imwrite(masked_image_path, result)

        base64_image = encode_image_to_base64(masked_image_path)
        data_url = f"data:image/png;base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": inference_model,
            "temperature": temperature,
            "messages": messages
        }

        response = requests.post(url, headers=headers, json=payload)
        annotations[key] = response.json()['choices'][0]['message']['content']

        # Save after each success
        with open(annotations_file, "w") as f:
            json.dump(annotations, f, indent=2)

    except Exception as e:
        print(f"Error processing key {key}: {e}")
        continue
