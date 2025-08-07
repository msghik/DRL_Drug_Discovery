"""
Attention-Augmented LSTM Predictor for SMILES Property Prediction

This module implements an enhanced predictive model that incorporates self-attention
mechanisms to better focus on chemically meaningful substructures in SMILES strings.

Architecture:
1. Embedding Layer: Converts SMILES tokens to dense vectors
2. Bi-directional LSTM Layer: Captures forward and backward dependencies
3. Self-Attention Layer: Aggregates contextual information across the sequence
4. Dense Layers: Final prediction layers

The attention mechanism allows the model to dynamically focus on important parts
of the molecular structure (e.g., rings, functional groups) that contribute most
to the predicted property.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import math


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence processing.
    
    This layer computes attention weights for each position in the sequence,
    allowing the model to focus on the most relevant parts for prediction.
    """
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        """
        Initialize self-attention layer.
        
        Parameters
        ----------
        hidden_size: int
            Hidden dimension size
        num_heads: int
            Number of attention heads
        dropout: float
            Dropout probability
        """
        super(SelfAttention, self).__init__()
        
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        """
        Forward pass of self-attention.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor [batch_size, seq_len, hidden_size]
        mask: torch.Tensor, optional
            Padding mask [batch_size, seq_len]
            
        Returns
        -------
        output: torch.Tensor
            Attention-weighted output [batch_size, seq_len, hidden_size]
        attention_weights: torch.Tensor
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Residual connection
        residual = x
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # Output projection
        output = self.out_proj(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer for sequence aggregation.
    
    This layer computes a weighted average of the sequence representations,
    where the weights are learned to focus on the most important positions.
    """
    
    def __init__(self, hidden_size):
        """
        Initialize attention pooling layer.
        
        Parameters
        ----------
        hidden_size: int
            Hidden dimension size
        """
        super(AttentionPooling, self).__init__()
        
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        """
        Forward pass of attention pooling.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor [batch_size, seq_len, hidden_size]
        mask: torch.Tensor, optional
            Padding mask [batch_size, seq_len]
            
        Returns
        -------
        pooled: torch.Tensor
            Pooled representation [batch_size, hidden_size]
        attention_weights: torch.Tensor
            Attention weights [batch_size, seq_len]
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        return pooled, attention_weights


class AttentionAugmentedLSTMPredictor(nn.Module):
    """
    Attention-augmented LSTM predictor for SMILES property prediction.
    
    This model combines bidirectional LSTM with self-attention mechanisms
    to better capture chemically relevant patterns in molecular structures.
    """
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_hidden_size=256, 
                 num_lstm_layers=2, num_attention_heads=8, num_dense_layers=2,
                 dense_hidden_size=512, output_size=1, dropout=0.1):
        """
        Initialize the attention-augmented LSTM predictor.
        
        Parameters
        ----------
        vocab_size: int
            Size of the vocabulary (number of unique SMILES tokens)
        embedding_dim: int
            Dimension of token embeddings
        lstm_hidden_size: int
            Hidden size of LSTM layers
        num_lstm_layers: int
            Number of LSTM layers
        num_attention_heads: int
            Number of self-attention heads
        num_dense_layers: int
            Number of dense layers after attention
        dense_hidden_size: int
            Hidden size of dense layers
        output_size: int
            Output dimension (1 for regression, num_classes for classification)
        dropout: float
            Dropout probability
        """
        super(AttentionAugmentedLSTMPredictor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        # Self-attention layer
        lstm_output_size = lstm_hidden_size * 2  # Bidirectional
        self.self_attention = SelfAttention(
            hidden_size=lstm_output_size,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(lstm_output_size)
        
        # Dense layers
        dense_layers = []
        input_size = lstm_output_size
        
        for i in range(num_dense_layers):
            dense_layers.append(nn.Linear(input_size, dense_hidden_size))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            input_size = dense_hidden_size
        
        # Output layer
        dense_layers.append(nn.Linear(input_size, output_size))
        
        self.dense_layers = nn.Sequential(*dense_layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, sequences, lengths=None, return_attention=False):
        """
        Forward pass of the model.
        
        Parameters
        ----------
        sequences: torch.Tensor
            Input sequences [batch_size, max_seq_len]
        lengths: torch.Tensor, optional
            Actual lengths of sequences [batch_size]
        return_attention: bool
            Whether to return attention weights
            
        Returns
        -------
        predictions: torch.Tensor
            Model predictions [batch_size, output_size]
        attention_info: dict, optional
            Attention weights if return_attention=True
        """
        batch_size, seq_len = sequences.size()
        
        # Create padding mask
        if lengths is not None:
            mask = torch.zeros(batch_size, seq_len, device=sequences.device)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1
        else:
            # Assume no padding if lengths not provided
            mask = (sequences != 0).float()
        
        # Embedding
        embedded = self.embedding(sequences)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM
        if lengths is not None:
            # Pack sequences for efficiency
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_lstm_out, (hidden, cell) = self.lstm(packed_embedded)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_lstm_out, batch_first=True
            )
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention
        attended_out, self_attention_weights = self.self_attention(lstm_out, mask)
        
        # Attention pooling
        pooled_out, pooling_attention_weights = self.attention_pooling(attended_out, mask)
        
        # Dense layers
        predictions = self.dense_layers(pooled_out)
        
        if return_attention:
            attention_info = {
                'self_attention_weights': self_attention_weights,
                'pooling_attention_weights': pooling_attention_weights
            }
            return predictions, attention_info
        
        return predictions


class AttentionAugmentedPredictor:
    """
    Wrapper class for the attention-augmented LSTM predictor.
    
    This class provides a similar interface to the original RNNPredictor
    but uses the enhanced attention-based architecture.
    """
    
    def __init__(self, vocab_size, model_params=None, ensemble_size=5, use_cuda=True):
        """
        Initialize the attention-augmented predictor.
        
        Parameters
        ----------
        vocab_size: int
            Size of the vocabulary
        model_params: dict, optional
            Model hyperparameters
        ensemble_size: int
            Number of models in the ensemble
        use_cuda: bool
            Whether to use GPU
        """
        self.vocab_size = vocab_size
        self.ensemble_size = ensemble_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Default model parameters
        default_params = {
            'embedding_dim': 128,
            'lstm_hidden_size': 256,
            'num_lstm_layers': 2,
            'num_attention_heads': 8,
            'num_dense_layers': 2,
            'dense_hidden_size': 512,
            'output_size': 1,
            'dropout': 0.1
        }
        
        if model_params:
            default_params.update(model_params)
        
        self.model_params = default_params
        
        # Create ensemble of models
        self.models = []
        for _ in range(ensemble_size):
            model = AttentionAugmentedLSTMPredictor(
                vocab_size=vocab_size,
                **self.model_params
            )
            if self.use_cuda:
                model = model.cuda()
            self.models.append(model)
    
    def tokenize_smiles(self, smiles_list, tokens, max_len=120):
        """
        Tokenize SMILES strings.
        
        Parameters
        ----------
        smiles_list: list
            List of SMILES strings
        tokens: str or list
            Vocabulary tokens
        max_len: int
            Maximum sequence length
            
        Returns
        -------
        sequences: torch.Tensor
            Tokenized sequences
        lengths: torch.Tensor
            Actual sequence lengths
        valid_indices: list
            Indices of valid SMILES
        """
        if isinstance(tokens, str):
            char_to_idx = {char: idx for idx, char in enumerate(tokens)}
        else:
            char_to_idx = {char: idx for idx, char in enumerate(tokens)}
        
        sequences = []
        lengths = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                # Canonicalize SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical_smiles = Chem.MolToSmiles(mol)
                    
                    # Tokenize
                    sequence = [char_to_idx.get(char, 0) for char in canonical_smiles[:max_len]]
                    
                    sequences.append(sequence)
                    lengths.append(len(sequence))
                    valid_indices.append(i)
            except:
                continue
        
        if not sequences:
            return None, None, []
        
        # Pad sequences
        max_seq_len = max(lengths)
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [0] * (max_seq_len - len(seq))
            padded_sequences.append(padded_seq)
        
        sequences_tensor = torch.LongTensor(padded_sequences)
        lengths_tensor = torch.LongTensor(lengths)
        
        if self.use_cuda:
            sequences_tensor = sequences_tensor.cuda()
            lengths_tensor = lengths_tensor.cuda()
        
        return sequences_tensor, lengths_tensor, valid_indices
    
    def predict(self, smiles_list, tokens, use_tqdm=False, return_attention=False):
        """
        Predict properties for SMILES strings.
        
        Parameters
        ----------
        smiles_list: list
            List of SMILES strings
        tokens: str or list
            Vocabulary tokens
        use_tqdm: bool
            Whether to show progress bar
        return_attention: bool
            Whether to return attention weights
            
        Returns
        -------
        canonical_smiles: list
            Canonicalized SMILES strings
        predictions: np.ndarray
            Predicted properties
        invalid_smiles: list
            Invalid SMILES strings
        attention_info: dict, optional
            Attention weights if return_attention=True
        """
        # Tokenize SMILES
        sequences, lengths, valid_indices = self.tokenize_smiles(smiles_list, tokens)
        
        if sequences is None:
            return [], [], smiles_list
        
        canonical_smiles = []
        invalid_smiles = []
        
        # Separate valid and invalid SMILES
        for i, smiles in enumerate(smiles_list):
            if i in valid_indices:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    canonical_smiles.append(Chem.MolToSmiles(mol))
                except:
                    invalid_smiles.append(smiles)
            else:
                invalid_smiles.append(smiles)
        
        # Predict with ensemble
        all_predictions = []
        all_attention_info = []
        
        iterator = tqdm(self.models) if use_tqdm else self.models
        
        for model in iterator:
            model.eval()
            with torch.no_grad():
                if return_attention:
                    preds, attention_info = model(sequences, lengths, return_attention=True)
                    all_attention_info.append(attention_info)
                else:
                    preds = model(sequences, lengths)
                
                all_predictions.append(preds.cpu().numpy())
        
        # Average predictions across ensemble
        predictions = np.mean(all_predictions, axis=0)
        
        if return_attention:
            # Average attention weights across ensemble
            avg_attention_info = {}
            for key in all_attention_info[0].keys():
                weights = torch.stack([info[key] for info in all_attention_info]).mean(dim=0)
                avg_attention_info[key] = weights.cpu().numpy()
            
            return canonical_smiles, predictions, invalid_smiles, avg_attention_info
        
        return canonical_smiles, predictions, invalid_smiles
    
    def train_model(self, train_data, val_data=None, epochs=100, batch_size=32, 
                   learning_rate=1e-3, patience=10):
        """
        Train the attention-augmented predictor.
        
        Parameters
        ----------
        train_data: tuple
            (smiles_list, labels, tokens)
        val_data: tuple, optional
            Validation data (smiles_list, labels, tokens)
        epochs: int
            Number of training epochs
        batch_size: int
            Batch size for training
        learning_rate: float
            Learning rate
        patience: int
            Early stopping patience
            
        Returns
        -------
        training_history: dict
            Training losses and metrics
        """
        smiles_list, labels, tokens = train_data
        
        # Tokenize training data
        sequences, lengths, valid_indices = self.tokenize_smiles(smiles_list, tokens)
        if sequences is None:
            raise ValueError("No valid SMILES found in training data")
        
        # Filter labels for valid sequences
        labels = np.array(labels)[valid_indices]
        labels_tensor = torch.FloatTensor(labels)
        if self.use_cuda:
            labels_tensor = labels_tensor.cuda()
        
        training_history = {'train_loss': [], 'val_loss': []}
        
        # Train each model in the ensemble
        for model_idx, model in enumerate(self.models):
            print(f"Training model {model_idx + 1}/{self.ensemble_size}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                
                # Create batches
                num_batches = (len(sequences) + batch_size - 1) // batch_size
                epoch_loss = 0.0
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(sequences))
                    
                    batch_sequences = sequences[start_idx:end_idx]
                    batch_lengths = lengths[start_idx:end_idx]
                    batch_labels = labels_tensor[start_idx:end_idx]
                    
                    optimizer.zero_grad()
                    
                    predictions = model(batch_sequences, batch_lengths)
                    loss = criterion(predictions.squeeze(), batch_labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / num_batches
                training_history['train_loss'].append(avg_train_loss)
                
                # Validation
                if val_data is not None:
                    val_loss = self._validate_model(model, val_data, tokens, criterion)
                    training_history['val_loss'].append(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
                    if val_data is not None:
                        print(f"Val Loss: {val_loss:.4f}")
        
        return training_history
    
    def _validate_model(self, model, val_data, tokens, criterion):
        """Validate a single model."""
        smiles_list, labels, _ = val_data
        
        sequences, lengths, valid_indices = self.tokenize_smiles(smiles_list, tokens)
        if sequences is None:
            return float('inf')
        
        labels = np.array(labels)[valid_indices]
        labels_tensor = torch.FloatTensor(labels)
        if self.use_cuda:
            labels_tensor = labels_tensor.cuda()
        
        model.eval()
        with torch.no_grad():
            predictions = model(sequences, lengths)
            loss = criterion(predictions.squeeze(), labels_tensor)
        
        return loss.item()
    
    def save_models(self, save_dir):
        """Save all models in the ensemble."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{save_dir}/model_{i}.pt")
        
        # Save model parameters
        import json
        with open(f"{save_dir}/model_params.json", 'w') as f:
            json.dump(self.model_params, f)
    
    def load_models(self, save_dir):
        """Load all models in the ensemble."""
        import json
        
        # Load model parameters
        with open(f"{save_dir}/model_params.json", 'r') as f:
            self.model_params = json.load(f)
        
        # Load models
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(f"{save_dir}/model_{i}.pt"))


def visualize_attention(smiles, attention_weights, tokens, save_path=None):
    """
    Visualize attention weights for a SMILES string.
    
    Parameters
    ----------
    smiles: str
        SMILES string
    attention_weights: np.ndarray
        Attention weights [seq_len]
    tokens: str or list
        Vocabulary tokens
    save_path: str, optional
        Path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Normalize attention weights
        attention_weights = attention_weights / attention_weights.max()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(max(10, len(smiles) * 0.5), 3))
        
        # Plot attention as a heatmap
        attention_matrix = attention_weights.reshape(1, -1)
        sns.heatmap(attention_matrix, 
                   xticklabels=list(smiles[:len(attention_weights)]),
                   yticklabels=['Attention'],
                   cmap='Reds',
                   cbar=True,
                   ax=ax)
        
        ax.set_title(f'Attention Weights for SMILES: {smiles}')
        ax.set_xlabel('SMILES Characters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib and seaborn are required for visualization")
        print("Install with: pip install matplotlib seaborn")
