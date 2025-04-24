# 2_generate_gpt4o_captions.py
import openai
import pandas as pd
import argparse
from tqdm import tqdm

# openai.api_key = "2c53e996ac264f8497cfbf6428d777e0"
import requests
import os

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
            {"role": "system", "content": "You are a helpful assistant skilled in describing vehicle damage."},
            {"role": "user", "content": prompt}
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

# def call_gpt4o(prompt):
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant skilled in describing vehicle damage."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.0
#     )
#     return response['choices'][0]['message']['content'].strip()

def refine_existing_captions(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    gpt4o_captions = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = f"Improve the following description to be more detailed and precise about the vehicle damage: '{row['caption']}'"
        caption = call_gpt4o(prompt)
        gpt4o_captions.append(caption)

    df['gpt4o_caption'] = gpt4o_captions
    df.to_csv(output_csv, index=False)
    print(f"✅ GPT-4o refined captions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    refine_existing_captions(args.input_csv, args.output_csv)