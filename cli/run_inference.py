import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import T5TokenizerFast
from src.model import T5FineTuner
from src.utils import load_data
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Run inference on new data with T5 model")
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR, help="Directory for dataset")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3], default=Config.MODE, help="Inference mode")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    model = T5FineTuner.load_from_checkpoint(args.checkpoint_path, tokenizer=tokenizer, mode=args.mode)

    _, _, test_dataloader = load_data(args.data_dir, tokenizer, args.batch_size)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            source_token_ids, source_mask, _, _, _, _ = batch
            source_token_ids = source_token_ids.to(model.device)
            source_mask = source_mask.to(model.device)

            pred_token_ids = model.generate(input_ids=source_token_ids, attention_mask=source_mask, max_length=Config.TARGET_MAX_LENGTH)
            predictions.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_token_ids])
    
    for i, pred in enumerate(predictions):
        print(f"Sample {i + 1}: {pred}")

if __name__ == "__main__":
    main()
