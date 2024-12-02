import click
import os
from os.path import join
import json
from src.text_encoder import CTCTextEncoder
from src.metrics.utils import calc_cer, calc_wer
import numpy as np


@click.command()
@click.option("--predictions_dir", type=str, required=True)
def main(predictions_dir):
    wers, cers = [], []

    for pred_json in os.listdir(predictions_dir):
        pred_json = join(predictions_dir, pred_json)
        with open(pred_json, 'r') as file:
            pred_json_dict = json.load(file)
            text = pred_json_dict["text"]
            pred_text = pred_json_dict["pred_text"]
            normalized_text = CTCTextEncoder.normalize_text(text)
            wers.append(calc_wer(normalized_text, pred_text))
            cers.append(calc_cer(normalized_text, pred_text))
    print("Your CER:", np.mean(cers))
    print("Your WER:", np.mean(wers))

if __name__ == "__main__":
    main()
