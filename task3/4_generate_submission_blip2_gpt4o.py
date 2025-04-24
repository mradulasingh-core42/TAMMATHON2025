import openai
import pandas as pd
import argparse
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm
import os
import requests

CORE42_API_KEY = "2c53e996ac264f8497cfbf6428d777e0" # or paste your key here as a string
API_URL = "https://api.core42.ai/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

def call_gpt4o(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CORE42_API_KEY}"
    }

    data = {
        "model": "gpt-4o",
        "messages": [
            # {"role": "system", "content": "You are a helpful assistant skilled in providing vehicle damage details in a very precise manner. Always start your answer with 'Car image with' phrase followed by the precise damages or issues in the car for example: 'shattered glass', 'dent', 'crack', 'scratch','broken lamp', 'flat tire' etc etc. The car can have multiple issues. If car has more than one issues, it should be comma separated for example: 'Car image with dent, scratch, flat tire.'. The responses should be precise as illustrated in above example responses."},
            # {"role": "user", "content": prompt}
            # { "role": "system", "content": 'You are an expert assistant in identifying and listing vehicle damages or issues based on provided damaged car image data.  Always start your response with the phrase "Car image with", followed by a precise, comma-separated list of damages or issues in the car. Examples of issues include: "shattered glass," "dent," "crack," "scratch," "broken lamp," "flat tire," etc. If the car has multiple issues, list them in a comma-separated format (e.g., "Car image with dent, scratch, flat tire."). Ensure responses are concise, accurate, and formatted exactly as shown in the examples'},{"role": "user", "content": prompt}
            
            { "role": "system", "content": 'You are an expert assistant in identifying and listing vehicle damages or issues based on provided damaged car image data.  You can identify any kind of damages whether it is minor or major car damage issue. Always start your response with the phrase "Car image with", followed by a precise, comma-separated list of damages or issues in the car. Examples of issues include: "shattered glass," "dent," "crack," "scratch," "broken lamp," "flat tire," etc. If the car has multiple issues, list them in a comma-separated format (e.g., "Car image with dent, scratch, flat tire."). Ensure responses are concise, accurate, and formatted exactly as shown in the examples'},{"role": "user", "content": prompt}
            
        ],
        "temperature": 0.0,
        "n": 1,
        "stream": False,
        "max_tokens": 4096,
        "user": "",
        "logprobs": False
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return "Error generating caption"   

def generate_submission(test_csv, test_img_dir, output_csv):
    df = pd.read_csv(test_csv)

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")

    final_captions = []
    blip2_captions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(test_img_dir, row['file_name'])
        raw_image = Image.open(image_path).convert("RGB")

        inputs = processor(images=raw_image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs)
        blip2_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        prompt = f"Refine the following description to be more precise about the vehicle damages: '{blip2_caption}'"
        final_caption = call_gpt4o(prompt)

        final_captions.append(final_caption)
        blip2_captions.append(blip2_caption)

    submission_df = pd.DataFrame({
        'id': df['id'],
        'file_name': df['file_name'],
        'prediction': final_captions,
        'blip2_caption':blip2_captions
    })
    submission_df.to_csv(output_csv, index=False)
    print(f"✅ Submission CSV saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--test_img_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    generate_submission(args.test_csv, args.test_img_dir, args.output_csv)