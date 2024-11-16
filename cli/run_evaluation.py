import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from src.evaluate import evaluate_model
from transformers import T5TokenizerFast
from src.model import T5FineTuner
from src.utils import load_data
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Evaluate T5 model on test data")
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR, help="Directory for test dataset")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=Config.MODE, help="Evaluation mode")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    model = T5FineTuner.load_from_checkpoint(args.checkpoint_path, tokenizer=tokenizer, mode=args.mode)

    _, _, test_dataloader = load_data(args.data_dir, tokenizer, args.batch_size)

    bleu_score = evaluate_model(model, test_dataloader)
    print(f"BLEU Score on test set: {bleu_score}")

if __name__ == "__main__":
    main()
