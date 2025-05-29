class Config:
    DATA_DIR = "./data"
    BATCH_SIZE = 2 
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 45
    WARMUP_EPOCHS = 5
    TARGET_MAX_LENGTH = 600
    MODE = 0  # Default to standard translation mode; change this to experiment with other modes
    ACCUMULATE_GRAD_BATCHES = 4  # This defines how many batches to accumulate gradients for
    PATIENCE = 3  # Number of epochs with no improvement after which training will be stopped
    CHECKPOINT_PATH = "PATH/FILE_NAME"  # Update this
    TEST_DATA_PATH = "preprocessed_files/test/preprocessed_test.pt"
