import os
import json
from sentence_transformers import SentenceTransformer
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import torch.nn.functional as F

# Paths
os.makedirs("data/embeddings", exist_ok=True)

images_dir = "data/images"
output_path = "data/embeddings/embeddings_model.json"
with open('data/annotations/annotations_gpt-4o.json', 'r') as f:
    gpt_annotations = json.load(f)
with open('data/annotations/annotations_qwen2.5.json', 'r') as f:
    qwen_annotations = json.load(f)


# Get all keys
keys = list(gpt_annotations.keys())

# Load models
sentence_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
clip_model = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k")

embeddings = {}
for key in keys:
    openai_text = gpt_annotations[key]
    qwen_text = qwen_annotations[key]

    image_path = os.path.join(images_dir, f"image_{key}.png")

    if not os.path.exists(image_path):
        continue

    # Sentence embedding
    sent_emb = sentence_model.encode([openai_text, qwen_text], convert_to_tensor=True)
    sent_emb = F.layer_norm(sent_emb, normalized_shape=(sent_emb.shape[1],))
    sent_emb = sent_emb[:, :512]
    sent_emb = F.normalize(sent_emb, p=2, dim=1)

    # Image embedding
    clip_emb = clip_model.embed_image([image_path])[0]

    embeddings[key] = {
        "clip": clip_emb,
        "gpt4o": sent_emb[0].tolist(),
        "qwen2.5": sent_emb[1].tolist()
    }

# Save embeddings
with open(output_path, "w") as f:
    json.dump(embeddings, f, indent=2)
