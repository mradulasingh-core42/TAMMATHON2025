# 1_extract_blip_features.py
import os
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


def generate_blip2_captions(csv_path, image_dir, output):
    df = pd.read_csv(csv_path)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")

    blip2_captions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(image_dir, row['file_name'])
        raw_image = Image.open(image_path).convert('RGB')

        inputs = processor(images=raw_image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        blip2_captions.append(caption)

    df['blip2_caption'] = blip2_captions
    df.to_csv(output, index=False)
    print(f"✅ BLIP-2 captions saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file with image names")
    parser.add_argument("--image_dir", required=True, help="Folder with images")
    parser.add_argument("--output", required=True, help="Path to save output CSV")
    args = parser.parse_args()

    generate_blip2_captions(args.csv, args.image_dir, args.output)
