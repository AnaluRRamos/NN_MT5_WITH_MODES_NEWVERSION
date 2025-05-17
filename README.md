

# Entity-Aware NMT for Biomedical Texts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]

A neural machine-translation system based on mT5 that incorporates biomedical named-entity information to improve translation fidelity on domain-specific terms.  

## Features

- 🔬 **Entity-aware encoder**: highlights genes, chemicals, diseases, etc.  
- ⚖️ **Dynamic loss weighting**: penalizes mistakes on entities more heavily.  
- 📊 Includes scripts for BLEU, CHRF, METEOR, ROUGE evaluation.  

## Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  
3. [Configuration](#configuration)  
4. [Evaluation](#evaluation)  
5. [Citation](#citation)  
6. [License](#license)  
7. [Contact](#contact)  

## Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/AnaluRRamos/NN_MT5_WITH_MODES_NEWVERSION/
   cd biomedical-nmt-entity-aware



To change the mode, go to the config.py and change manually.
The data files are just to heavy therefore they will be shared: 
