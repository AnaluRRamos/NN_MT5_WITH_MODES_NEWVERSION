import sys
import os
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model
from src.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parse command-line arguments to allow overriding default configuration."""
    parser = argparse.ArgumentParser(description="Run training for MT5FineTuner model.")
    parser.add_argument('--train_data', type=str, default="preprocessed_train.pt", help='Path to preprocessed training data.')
    parser.add_argument('--val_data', type=str, default="preprocessed_val.pt", help='Path to preprocessed validation data.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs.')
    parser.add_argument('--mode', type=int, help='Mode for model configuration.')
    return parser.parse_args()

def override_config_with_args(args):
    """Override configuration values with parsed arguments if provided."""
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
        logging.info(f"Overriding batch size to: {Config.BATCH_SIZE}")
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
        logging.info(f"Overriding learning rate to: {Config.LEARNING_RATE}")
    if args.max_epochs:
        Config.MAX_EPOCHS = args.max_epochs
        logging.info(f"Overriding max epochs to: {Config.MAX_EPOCHS}")
    if args.mode:
        Config.MODE = args.mode
        logging.info(f"Overriding mode to: {Config.MODE}")

def main():
    args = parse_args()
    override_config_with_args(args)

    logging.info("Starting training with updated configuration.")
    train_model()

if __name__ == "__main__":
    main()
