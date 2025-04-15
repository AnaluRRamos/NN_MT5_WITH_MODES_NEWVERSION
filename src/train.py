
import torch
import pytorch_lightning as pl
from transformers import MT5TokenizerFast
from src.model import MT5FineTuner
from src.utils import load_data
from src.config import Config
import logging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CometLogger

logging.basicConfig(level=logging.INFO)

def train_model():
    logging.info("Loading tokenizer...")
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base")

    logging.info("Loading preprocessed training and validation data...")
    train_dataloader, val_dataloader = load_data(
        preprocessed_train="preprocessed_files/train/preprocessed_train.pt",
        preprocessed_val="preprocessed_files/val/preprocessed_val.pt",
        batch_size=Config.BATCH_SIZE
    )

    logging.info("Initializing the model...")
    model = MT5FineTuner(
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        learning_rate=Config.LEARNING_RATE,
        target_max_length=Config.TARGET_MAX_LENGTH,
        mode=Config.MODE  # Set to 1 to use the NE tag loss, for example.
    )

    logging.info("Starting training...")

    # CHANGED: Added ModelCheckpoint callback to save checkpoints in the desired directory.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="data/checkpoints",  
        filename="t5_finetuner-{epoch:02d}-{val_loss:.2f}-{val_bleu:.2f}",
        save_top_k=2,
        monitor="val_bleu",
        mode="max"
    )


    # CHANGED: Comet ML logger 13/03
    comet_logger = CometLogger(api_key="HvL5dz50hej2GlMfczDmW99Hk", project_name="your_project_name")
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu',
        devices=1,
        #devices=torch.cuda.device_count(),
        #strategy="ddp_find_unused_parameters_true",
        precision=32,  # Consider switching to 16 for mixed precision if needed.
        accumulate_grad_batches=Config.ACCUMULATE_GRAD_BATCHES,
        callbacks=[checkpoint_callback],  # CHANGED: Added the checkpoint callback here.
        logger=comet_logger,
        log_every_n_steps=50  # Logs metrics every 50 steps changed 23/03
    )

    trainer.fit(model)
    logging.info("Training completed.")
    # CHANGED: Log the best checkpoint path (only rank 0 will output this in DDP).
    logging.info(f"Best model path: {checkpoint_callback.best_model_path}")
    return model

if __name__ == "__main__":
    train_model()
