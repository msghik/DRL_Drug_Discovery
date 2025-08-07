# ReLeaSE: Reinforcement Learning for Structural Evolution (Enhanced)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

## Overview

ReLeaSE is a Python package for **de novo molecular design** using **deep reinforcement learning**. This enhanced version features transformer-based generative models and attention-augmented predictive components, providing state-of-the-art performance for molecular generation and property optimization.

### Key Features

- 🧬 **Transformer-Based Generation**: GPT-style and BART-style transformers for molecular SMILES generation
- 🎯 **Attention-Augmented Prediction**: BiLSTM models with self-attention mechanisms for property prediction
- ⚡ **Reinforcement Learning**: Policy gradient optimization for biasing molecular generation toward desired properties
- 🔧 **Modular Architecture**: Easy-to-use components that can be combined flexibly
- 🚀 **GPU Acceleration**: Full CUDA support for faster training and inference
- 📊 **Comprehensive Evaluation**: Built-in molecular property prediction and validation

## Architecture Overview

### Generative Models
- **GPT-Style Transformers**: Autoregressive molecular generation with self-attention
- **BART-Style Transformers**: Encoder-decoder architecture for molecular optimization
- **Legacy Stack-RNN**: Original RNN-based generator with augmented memory

### Predictive Models
- **Attention-Augmented BiLSTM**: Enhanced LSTM with self-attention for property prediction
- **Vanilla QSAR**: Traditional machine learning models with molecular fingerprints
- **RNN Predictor**: Deep learning models for molecular property estimation

### Reinforcement Learning
- **Policy Gradient**: Optimize molecular generation toward desired properties
- **Custom Reward Functions**: Flexible reward design for various optimization objectives
- **Multi-Objective Optimization**: Support for multiple property targets

## Installation

### Prerequisites

```bash
# Required dependencies
pip install torch>=1.9.0
pip install numpy>=1.19.0
pip install scipy>=1.6.0
pip install scikit-learn>=0.24.0
pip install rdkit-pypi>=2021.3.1
pip install tqdm>=4.60.0
```

### Install from Source

```bash
git clone https://github.com/isayev/ReLeaSE.git
cd ReLeaSE
pip install -e .
```

## Quick Start

### 1. Basic Molecular Generation

```python
from release.data import GeneratorData
from release.transformer_generator import GPTStyleGenerator

# Load training data
data = GeneratorData('data/chembl_22_clean_1576904_sorted_std_final.smi')

# Create transformer generator
generator = GPTStyleGenerator(
    vocab_size=data.n_characters,
    d_model=512,
    nhead=8,
    num_layers=6
)

# Train the model
losses = generator.fit(data, n_iterations=1000)

# Generate new molecules
for _ in range(10):
    smiles = generator.evaluate(data)
    print(smiles)
```

### 2. Property Prediction with Attention

```python
from release.attention_predictor import AttentionAugmentedPredictor
from release.data import PredictorData

# Load property data
pred_data = PredictorData('data/logP_labels.csv')

# Create attention-augmented predictor
predictor = AttentionAugmentedPredictor(
    vocab_size=pred_data.vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    attention_dim=128,
    num_layers=2
)

# Train the predictor
predictor.fit(pred_data)

# Predict properties
smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
predictions = predictor.predict(smiles_list)
print(predictions)
```

### 3. Reinforcement Learning Optimization

```python
from release.reinforcement import Reinforcement

# Define custom reward function
def logp_reward(smiles, predictor, target_logp=2.0):
    """Reward function targeting specific logP value."""
    try:
        pred_logp = predictor.predict([smiles])[0]
        return 1.0 - abs(pred_logp - target_logp) / 5.0
    except:
        return 0.0

# Setup reinforcement learning
rl_optimizer = Reinforcement(
    generator=generator,
    predictor=predictor,
    get_reward=logp_reward
)

# Run optimization
for epoch in range(100):
    reward, loss = rl_optimizer.policy_gradient(
        data=data,
        n_batch=32,
        target_logp=2.0
    )
    print(f"Epoch {epoch}: Reward={reward:.3f}, Loss={loss:.3f}")
```

## Model Architectures

### Transformer Generators

#### GPT-Style Generator
```python
from release.transformer_generator import GPTStyleGenerator

generator = GPTStyleGenerator(
    vocab_size=50,          # SMILES vocabulary size
    d_model=512,            # Model dimension
    nhead=8,                # Number of attention heads
    num_layers=6,           # Number of transformer layers
    max_len=200,            # Maximum sequence length
    dropout=0.1             # Dropout rate
)
```

#### BART-Style Generator
```python
from release.transformer_generator import BARTStyleGenerator

generator = BARTStyleGenerator(
    vocab_size=50,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,   # Encoder layers
    num_decoder_layers=6,   # Decoder layers
    max_len=200,
    dropout=0.1
)
```

### Attention-Augmented Predictor

```python
from release.attention_predictor import AttentionAugmentedPredictor

predictor = AttentionAugmentedPredictor(
    vocab_size=50,
    embedding_dim=128,      # Character embedding dimension
    hidden_dim=256,         # LSTM hidden dimension
    attention_dim=128,      # Attention mechanism dimension
    num_layers=2,           # Number of LSTM layers
    num_classes=1,          # Regression (1) or classification (>1)
    dropout=0.1
)
```

## Data Formats

### Training Data
SMILES strings in text files, one per line:
```
CCO
CC(=O)O
c1ccccc1
...
```

### Property Data
CSV format with SMILES and property values:
```csv
SMILES,LogP,MW,TPSA
CCO,−0.31,46.07,20.23
CC(=O)O,−0.17,60.05,37.30
c1ccccc1,2.13,78.11,0.00
...
```

## Advanced Usage

### Custom Reward Functions

```python
def multi_objective_reward(smiles, predictor, targets):
    """Multi-objective optimization reward."""
    try:
        props = predictor.predict([smiles])
        logp, mw, tpsa = props[0]
        
        # Lipinski's Rule of Five constraints
        logp_score = 1.0 if -0.4 <= logp <= 5.6 else 0.0
        mw_score = 1.0 if mw <= 500 else 0.0
        tpsa_score = 1.0 if tpsa <= 140 else 0.0
        
        return (logp_score + mw_score + tpsa_score) / 3.0
    except:
        return 0.0
```

### Model Ensembles

```python
# Create ensemble of predictors
from release.predictor import VanillaQSAR
from sklearn.ensemble import RandomForestRegressor

ensemble_predictor = VanillaQSAR(
    model_instance=RandomForestRegressor,
    model_params={'n_estimators': 100, 'random_state': 42},
    ensemble_size=5,
    normalization=True
)
```

### SMILES Augmentation

```python
from release.smiles_enumerator import SmilesEnumerator

# Enable SMILES augmentation for training
augmenter = SmilesEnumerator(canonical=False, enum=True)
generator.fit(data, n_iterations=1000, augment=True)
```

### Running Tests
```bash
pytest tests/
python -m pytest tests/ --cov=release
```
