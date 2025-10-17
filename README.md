# Entity-Aware Neural Machine Translation for Biomedical Texts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)

A neural machine translation (NMT) system based on **mT5** that incorporates **biomedical named-entity information** to improve translation fidelity on domain-specific terms.

This repository accompanies the paper *“An Investigation of Entity-Aware Neural Machine Translation for Biomedical Texts”* (submitted to **LREC 2026**) and serves as a **language resource** for reproducible biomedical translation research.

---

##  Features

- **Entity-aware encoder** highlighting genes, chemicals, diseases, and other biomedical terms.
- **Two operational modes**:
  - `MODE 0` → fine-tuned model (no NER)
  - `MODE 1` → entity-aware model with NE embeddings and weighted loss
- **Dynamic loss weighting** that penalizes mistakes on biomedical entities more heavily.
- Scripts for **BLEU**, **CHRF**, **METEOR**, and **ROUGE** evaluation.
- Includes a **Manual for Human Evaluation** with examples and error annotation guidelines.

---

##  Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Preprocess](#preprocess)
4. [Train](#train)
5. [Test](#test)
6. [Evaluation](#evaluation)
7. [Citation](#citation)
8. [License](#license)
9. [Contact](#contact)

---

## Installation

```bash
git clone https://anonymous.4open.science/r/NN_MT5_WITH_MODES_NEWVERSION/
cd NN_MT5_WITH_MODES_NEWVERSION
pip install -r requirements.txt
```


## Usage

The project includes several scripts inside the cli/ and src/ directories.
All experiments can be reproduced using these components.

## Preprocess

First step is to preprocess the data — in this case, tokenize and create the labels for NE (Named Entities) for biomedical terms.
For that, use the preprocess_data.py (inside src/).
Inside the preprocess, you will see code where we read the text and tokenize for mT5 and also create the entity embeddings that will be used for MODE 1 in the model.py.

## Train

After preprocessing, let’s train the model!
But first, check the config.py inside src/ and make sure which MODE you want to use — 0 or 1.
MODE 0: standard translation (fine-tuned baseline)
MODE 1: entity-aware translation using NE embeddings and dynamic loss weighting
The idea here is to make the model avoid mistakes with important biomedical terms.
You can also change other hyperparameters.
Perfect! If you checked all of that, you can train the model by running the training script inside cli/:

```python cli/run_train.py```


## Test
Now, if you want to use the checkpoint generated during training, just check the path and then run:

```python cli/run_test.py```

This will produce translation outputs and compute automatic evaluation metrics.

## Evaluation

This project is part of a full evaluation framework for biomedical translation.
During the training and test phases, you will obtain automatic metrics such as BLEU, CHRF, METEOR, and ROUGE to evaluate translation performance.
If you want to go even further, you can check the Manual for Human Evaluation, which details how to manually evaluate outputs.
This manual provides a detailed way to compare translations and classify mistakes by type and severity.
An Excel file is also provided as an example of how this process can be done.

## Citation
If you use this resource in your research, please cite:
An Investigation of Entity-Aware Neural Machine Translation for Biomedical Texts.

📜 License
Distributed under the MIT License.

📧 Contact
For questions please reach out.
