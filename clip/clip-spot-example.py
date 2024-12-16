import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Models: {clip.available_models()}")
model, preprocess = clip.load("ViT-B/32", device=device)

img_dir_path = "/home/sobits/catkin_ws/src/images"
# categories = ['snack', 'shoes', 'chair', 'blanket', 'green tape', 'pet bottle', 'pringles', 'pen case', 'banana', 'blue pen', 'black pen', 'yellow pen', 'red pen', 'locker', 'table', 'sofa']
categories = ['door', 'book']
#categories = ["ukulele"]
category_idx = 0

text = clip.tokenize(categories).to(device)

img_names = os.listdir(img_dir_path)

best_value = 0
best_image = None

for img_name in img_names:
    img_path = img_dir_path + "/" + img_name

    raw_image = Image.open(img_path)
    #raw_image.show()
    image = preprocess(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs = logits_per_image

        if probs[0][category_idx] > best_value:
            best_value = probs[0][category_idx]
            best_image = img_path


    category_probs = sorted([(categories[idx], probs[0][idx]) for idx in range(len(categories))], key=lambda x: x[1], reverse=True)
    print(f"Sorted categories with probabilities: {category_probs}")

if best_image:
    print(f"Best image is: {best_image}")
    raw_image = Image.open(best_image)
    save_path = "best_image.jpg"
    raw_image.save(save_path)
    print(f"Best image has been saved to {save_path}")
else:
    print("No best image found.")
