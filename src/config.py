class Config:
    DATA_DIR = "./data"
    BATCH_SIZE = 1  # change to 8 when possible
    LEARNING_RATE = 1e-5
    MAX_EPOCHS = 10
    TARGET_MAX_LENGTH = 32
    MODE = 0  # Default to standard translation mode; change this to experiment with other modes
    ACCUMULATE_GRAD_BATCHES = 4  # This defines how many batches to accumulate gradients for
    PATIENCE = 3  # Number of epochs with no improvement after which training will be stopped


"""T5 models need a slightly higher learning rate than the default one set 
in the Trainer when using the AdamW optimizer. Typically, 1e-4 and 3e-4 work 
well for most problems (classification, summarization, translation, 
question answering, question generation). Note that T5 was pre-trained using 
the AdaFactor optimizer."""