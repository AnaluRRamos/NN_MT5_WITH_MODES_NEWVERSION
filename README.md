biomedical_translation_project/
│
├── data/                              # Store dataset files here
│   ├── train
│   ├── val
│   └── test
│
├── src/                               # Main source code
│   ├── __init__.py
│   ├── preprocess.py                  # For text and entity preprocessing
│   ├── model.py                       # Lightning model with entity-aware modes
│   ├── loss_functions.py              # Custom loss functions for different modes
│   ├── train.py                       # Trainer setup and model training with Lightning
│   ├── evaluate.py                    # Evaluation functions and metrics
│   ├── utils.py                       # Utility functions (e.g., loading data)
│   ├── mode_config.py                 # Configuration and hyperparameters for each mode
│   └── config.py                      # General configuration and hyperparameters
│
├── cli/                               # Command-line interface scripts
│   ├── run_train.py                   # CLI for training the model with Lightning, mode selection
│   ├── run_inference.py               # CLI for inference on new data, supporting modes
│   └── run_evaluation.py              # CLI for evaluating the model on test data
│
├── notebooks/                         # Jupyter notebooks for experiments and EDA
│   ├── data_exploration.ipynb
│   └── model_experiments.ipynb
│
├── output/                            # Model outputs
│   ├── checkpoints/                   # Saved model checkpoints, organized by mode
│   ├── predictions/                   # Translation outputs by mode
│   └── logs/                          # Training and evaluation logs by mode
│
└── README.md                          # Project description and instructions

Make sure you use the correct enviroment:
source en_ner_env/bin/activate
Set in the terminal before runing: 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


To change the mode, go to the config.py and change manually.
The data files are just to heavy therefore they will be shared: 
