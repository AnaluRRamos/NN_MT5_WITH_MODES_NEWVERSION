import comet_ml
import torch
from transformers import MT5TokenizerFast
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.model import MT5FineTuner
from src.utils import MT5DatasetPreprocessed
from src.config import Config
#import wandb

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

    # Initialize WandB logger
    #wandb_logger = WandbLogger(project="my_project_name", name="test_run")
    comet_logger = CometLogger(api_key="HvL5dz50hej2GlMfczDmW99Hk", project_name="your_project_name")


    # Initialize Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        #logger=wandb_logger
        logger=comet_logger
    )

    # Run test
    trainer.test(model, dataloaders=test_dataloader)

    # Finish WandB run
    #wandb.finish()
