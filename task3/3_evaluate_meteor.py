# 3_evaluate_meteor.py
import pandas as pd
from nltk.translate.meteor_score import meteor_score
import argparse
import nltk
nltk.download('wordnet')


def evaluate_meteor(reference_csv, prediction_csv):
    ref_df = pd.read_csv(reference_csv)
    pred_df = pd.read_csv(prediction_csv)

    scores = []
    for _, ref_row in ref_df.iterrows():
        pred_row = pred_df[pred_df['file_name'] == ref_row['file_name']]
        if not pred_row.empty:
            ref = ref_row['caption']
            pred = pred_row.iloc[0]['gpt4o_caption']
            score = meteor_score([ref.split()], pred.split())  # Tokenize here
            scores.append(score)

    mean_score = sum(scores) / len(scores)
    print(f"🌟 Average METEOR Score: {mean_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_csv", required=True, help="CSV with reference captions")
    parser.add_argument("--pred_csv", required=True, help="CSV with GPT-4o predictions")
    args = parser.parse_args()

    evaluate_meteor(args.ref_csv, args.pred_csv)