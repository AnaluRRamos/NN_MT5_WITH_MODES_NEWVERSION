# run_train.py

import sys
import os
import argparse
import logging

# Add the parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parse command-line arguments to allow overriding default configuration."""
    parser = argparse.ArgumentParser(description="Run training for T5FineTuner model.")
    parser.add_argument('--data_dir', type=str, help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs.')
    parser.add_argument('--mode', type=int, help='Mode for model configuration.')
    return parser.parse_args()

def override_config_with_args(args):
    """Override configuration values with parsed arguments if provided."""
    if args.data_dir:
        Config.DATA_DIR = args.data_dir
        logging.info(f"Overriding data directory to: {Config.DATA_DIR}")
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
    # Parse command-line arguments
    args = parse_args()
    
    # Override config with command-line arguments, if any
    override_config_with_args(args)
    
    # Log final configuration
    logging.info("Starting the training process with the following configuration:")
    logging.info(f"Data Directory: {Config.DATA_DIR}")
    logging.info(f"Batch Size: {Config.BATCH_SIZE}")
    logging.info(f"Learning Rate: {Config.LEARNING_RATE}")
    logging.info(f"Max Epochs: {Config.MAX_EPOCHS}")
    logging.info(f"Mode: {Config.MODE}")

    # Run training
    try:
        train_model()
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        sys.exit(1)

    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()


