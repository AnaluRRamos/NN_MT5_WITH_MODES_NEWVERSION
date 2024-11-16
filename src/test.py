import os
import argparse
from src.utils import load_data
from src.model import T5FineTuner
from src.evaluate import evaluate_model

def load_test_data(test_path):
    with open(test_path, 'r') as f:
        data = f.readlines()
    return [line.strip() for line in data]

def test_model(model, test_data):
    predictions = []
    for text in test_data:
        prediction = model.translate(text)
        predictions.append(prediction)
    return predictions

def main(test_path, checkpoint_path, mode):
    test_data = load_test_data(test_path)
    # Load the model checkpoint without `mode` directly, then set mode separately
    model = T5FineTuner.load_from_checkpoint(checkpoint_path)
    model.mode = mode  # Set the mode after loading the model
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Test the model
    predictions = test_model(model, test_data)
    
    # Evaluate model with BLEU score
    bleu_score = evaluate_model(model, predictions, test_data)
    print(f"BLEU Score for mode {mode}: {bleu_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on the test dataset.")
    parser.add_argument('--test_path', type=str, default='data/test/medline_en2pt_en.txt', help="Path to the test file.")
    parser.add_argument('--checkpoint_path', type=str, default='output/checkpoints/best_model.ckpt', help="Path to the model checkpoint.")
    parser.add_argument('--mode', type=int, default=0, help="Mode for model evaluation.")
    args = parser.parse_args()
    main(args.test_path, args.checkpoint_path, args.mode)
