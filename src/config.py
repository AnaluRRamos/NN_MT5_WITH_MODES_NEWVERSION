class Config:
    DATA_DIR = "./data"
    BATCH_SIZE = 2 # change to 8 when possible
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 20
    WARMUP_EPOCHS = 5
    TARGET_MAX_LENGTH = 600
    MODE = 1  # Default to standard translation mode; change this to experiment with other modes
    ACCUMULATE_GRAD_BATCHES = 4  # This defines how many batches to accumulate gradients for
    PATIENCE = 3  # Number of epochs with no improvement after which training will be stopped
    CHECKPOINT_PATH = "/home/analu/NN_MT_T5_MODES_NEW_VERSION/data/checkpoints/t5_finetuner-epoch=18-val_loss=2.45-val_bleu=38.21.ckpt"  # Update this
    TEST_DATA_PATH = "preprocessed_files/test/preprocessed_test.pt"


"""T5 models need a slightly higher learning rate than the default one set 
in the Trainer when using the AdamW optimizer. Typically, 1e-4 and 3e-4 work 
well for most problems (classification, summarization, translation, 
question answering, question generation). Note that T5 was pre-trained using 
the AdaFactor optimizer."""
""" Try it with LR= 3e-4"""
""" One idea is to train different layers with different LR"""
