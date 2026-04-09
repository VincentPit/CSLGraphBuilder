import torch
import open_clip
from PIL import Image
import os
import numpy as np

# Load model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

image_embeddings = {}
img_dir = "images"

with torch.no_grad():
    for filename in os.listdir(img_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image = Image.open(os.path.join(img_dir, filename)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)
        image_feature = model.encode_image(image_tensor)
        image_embeddings[filename] = image_feature / image_feature.norm()  # normalize


text_prompts = ["a motor", "a circuit board", "a coiled cable", "tool box"]

with torch.no_grad():
    tokenized = tokenizer(text_prompts)
    text_features = model.encode_text(tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # normalize


# Convert image embeddings to a tensor
image_names = list(image_embeddings.keys())
image_tensors = torch.stack([image_embeddings[name] for name in image_names])

# Compute similarity (text → images)
similarity = (text_features @ image_tensors.T).cpu().numpy()  # shape: (num_texts, num_images)

# Print matches
for i, prompt in enumerate(text_prompts):
    best_idx = np.argmax(similarity[i])
    best_match = image_names[best_idx]
    print(f"'{prompt}' best matches with: {best_match} (score: {similarity[i][best_idx]:.3f})")