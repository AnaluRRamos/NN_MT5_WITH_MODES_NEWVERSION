
import comet_ml  
import torch
from transformers import MT5TokenizerFast
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from src.model import MT5FineTuner
from src.utils import MT5DatasetPreprocessed
from src.config import Config

def test_model(args):
    """Runs the testing process for the MT5FineTuner model."""
    
    # Load tokenizer
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base")
    
    # Load preprocessed test data
    test_dataset = MT5DatasetPreprocessed(args.test_data)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Load model from checkpoint
    model = MT5FineTuner.load_from_checkpoint(
        args.checkpoint,
        tokenizer=tokenizer,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=test_dataloader,
        learning_rate=Config.LEARNING_RATE,
        target_max_length=Config.TARGET_MAX_LENGTH,
        mode=args.mode
    )
    
    # Comet Logger (optional)
    comet_logger = CometLogger(api_key="HvL5dz50hej2GlMfczDmW99Hk", project_name="your_project_name")
    
    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=comet_logger
    )
    
    # Run test
    results = trainer.test(model, dataloaders=test_dataloader)

    print(f"Testing completed.")
    print(f"BLEU scores saved to: bleu_scores_mode_{args.mode}.txt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="path/to/test/data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    test_model(args)



