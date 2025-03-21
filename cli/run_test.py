
import sys
import os
import argparse
import logging


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config  
from src.test import test_model  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run testing for MT5FineTuner model.")
    parser.add_argument('--test_data', type=str, default=Config.TEST_DATA_PATH, help='Path to preprocessed test data.')
    parser.add_argument('--checkpoint', type=str, default=Config.CHECKPOINT_PATH, help='Path to trained model checkpoint.')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size for testing.')
    parser.add_argument('--mode', type=int, default=Config.MODE, help='Mode for model configuration.')
    return parser.parse_args()

def main():
    args = parse_args()

    logging.info(f"Starting testing with checkpoint: {args.checkpoint}")

    test_model(args) 

if __name__ == "__main__":
    main()

