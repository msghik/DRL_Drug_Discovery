"""
Transformer-based molecular generator for ReLeaSE framework.
Supports multiple transformer architectures:
- GPT-style autoregressive transformers
- BART/T5-style encoder-decoder models
- Pretrained molecular transformers (MolBART, ChemGPT, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
import time
from tqdm import trange
import numpy as np

from utils import time_since
from smiles_enumerator import SmilesEnumerator


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class GPTStyleGenerator(nn.Module):
    """
    GPT-style autoregressive transformer for molecular generation.
    Similar to ChemGPT/MolGPT architectures.
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 max_len=200, dropout=0.1, use_cuda=None,
                 optimizer_instance=torch.optim.AdamW, lr=1e-4):
        """
        Constructor for GPT-style transformer generator.
        
        Parameters
        ----------
        vocab_size: int
            Size of the vocabulary (number of unique tokens)
        d_model: int
            Dimension of the model (embedding size)
        nhead: int
            Number of attention heads
        num_layers: int
            Number of transformer layers
        max_len: int
            Maximum sequence length
        dropout: float
            Dropout probability
        use_cuda: bool
            Whether to use GPU
        optimizer_instance: torch.optim
            Optimizer class
        lr: float
            Learning rate
        """
        super(GPTStyleGenerator, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer decoder layers (autoregressive)
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # CUDA setup
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = optimizer_instance(self.parameters(), lr=lr, weight_decay=1e-5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_causal_mask(self, size):
        """Create causal (lower triangular) mask for autoregressive generation."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        if self.use_cuda:
            mask = mask.cuda()
        return mask
    
    def forward(self, src, tgt):
        """
        Forward pass for training.
        
        Parameters
        ----------
        src: torch.Tensor
            Source sequence (input tokens)
        tgt: torch.Tensor
            Target sequence (shifted input for teacher forcing)
        
        Returns
        -------
        output: torch.Tensor
            Predicted token probabilities
        """
        seq_len = tgt.size(1)
        
        # Create causal mask
        tgt_mask = self._create_causal_mask(seq_len)
        
        # Embeddings and positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Transformer forward pass
        output = self.transformer(tgt_emb, src_emb, tgt_mask=tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def generate(self, data, prime_str='<', end_token='>', max_len=100, temperature=1.0):
        """
        Generate a new sequence using autoregressive sampling.
        
        Parameters
        ----------
        data: GeneratorData
            Data object containing vocabulary mappings
        prime_str: str
            Starting string
        end_token: str
            Token to end generation
        max_len: int
            Maximum generation length
        temperature: float
            Sampling temperature (higher = more diverse)
        
        Returns
        -------
        generated_sequence: str
            Generated molecular sequence
        """
        self.eval()
        with torch.no_grad():
            # Initialize with prime string
            generated = prime_str
            tokens = [data.char2idx[c] for c in prime_str]
            
            for _ in range(max_len):
                # Convert to tensor
                input_tensor = torch.LongTensor([tokens]).unsqueeze(0)
                if self.use_cuda:
                    input_tensor = input_tensor.cuda()
                
                # Forward pass
                src_emb = self.embedding(input_tensor) * math.sqrt(self.d_model)
                src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
                
                seq_len = input_tensor.size(1)
                tgt_mask = self._create_causal_mask(seq_len)
                
                output = self.transformer(src_emb, src_emb, tgt_mask=tgt_mask)
                output = self.output_projection(output)
                
                # Sample next token
                logits = output[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Convert to character and add to sequence
                next_char = data.all_characters[next_token]
                generated += next_char
                tokens.append(next_token)
                
                # Check for end token
                if next_char == end_token:
                    break
            
        self.train()
        return generated
    
    def train_step(self, inp, target):
        """
        Single training step.
        
        Parameters
        ----------
        inp: torch.Tensor
            Input sequence
        target: torch.Tensor
            Target sequence
        
        Returns
        -------
        loss: float
            Training loss
        """
        self.optimizer.zero_grad()
        
        # Add batch dimension if needed
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # Forward pass (use inp as both src and tgt for autoregressive training)
        output = self.forward(inp, inp)
        
        # Compute loss
        loss = self.criterion(output.view(-1, self.vocab_size), target.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def fit(self, data, n_iterations, all_losses=[], print_every=100,
            plot_every=10, augment=False):
        """
        Train the model.
        
        Parameters
        ----------
        data: GeneratorData
            Training data
        n_iterations: int
            Number of training iterations
        all_losses: list
            List to store losses
        print_every: int
            Print frequency
        plot_every: int
            Loss recording frequency
        augment: bool
            Whether to use SMILES augmentation
        
        Returns
        -------
        all_losses: list
            Training losses
        """
        start = time.time()
        loss_avg = 0
        
        if augment:
            smiles_augmentation = SmilesEnumerator()
        else:
            smiles_augmentation = None
        
        for epoch in trange(1, n_iterations + 1, desc='Training Transformer...'):
            inp, target = data.random_training_set(smiles_augmentation)
            loss = self.train_step(inp, target)
            loss_avg += loss
            
            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,
                                              epoch / n_iterations * 100, loss))
                print(self.generate(data=data, prime_str='<', max_len=100), '\n')
            
            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
        
        return all_losses
    
    def evaluate(self, data, prime_str='<', end_token='>', predict_len=100):
        """
        Generate a new sequence (compatible with original interface).
        
        Parameters
        ----------
        data: GeneratorData
            Data object
        prime_str: str
            Starting string
        end_token: str
            End token
        predict_len: int
            Maximum length
        
        Returns
        -------
        generated: str
            Generated sequence
        """
        return self.generate(data, prime_str, end_token, predict_len)
    
    def load_model(self, path):
        """Load model weights."""
        weights = torch.load(path, map_location='cpu' if not self.use_cuda else None)
        self.load_state_dict(weights)
    
    def save_model(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def change_lr(self, new_lr):
        """Update learning rate."""
        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr


class BARTStyleGenerator(nn.Module):
    """
    BART/T5-style encoder-decoder transformer for molecular generation.
    Useful for tasks like molecular optimization, translation, etc.
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, max_len=200, dropout=0.1, use_cuda=None,
                 optimizer_instance=torch.optim.AdamW, lr=1e-4):
        """
        Constructor for BART-style encoder-decoder transformer.
        
        Parameters similar to GPTStyleGenerator but with separate encoder/decoder layers.
        """
        super(BARTStyleGenerator, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # CUDA setup
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = optimizer_instance(self.parameters(), lr=lr, weight_decay=1e-5)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_causal_mask(self, size):
        """Create causal mask for decoder."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        if self.use_cuda:
            mask = mask.cuda()
        return mask
    
    def forward(self, src, tgt):
        """
        Forward pass for encoder-decoder architecture.
        
        Parameters
        ----------
        src: torch.Tensor
            Source sequence
        tgt: torch.Tensor
            Target sequence
        
        Returns
        -------
        output: torch.Tensor
            Decoded output
        """
        tgt_mask = self._create_causal_mask(tgt.size(1))
        
        # Encoder
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        memory = self.encoder(src_emb)
        
        # Decoder
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def generate(self, data, prime_str='<', end_token='>', max_len=100, temperature=1.0):
        """Generate sequence using encoder-decoder architecture."""
        self.eval()
        with torch.no_grad():
            # Encode the prime string
            prime_tokens = [data.char2idx[c] for c in prime_str]
            src_tensor = torch.LongTensor([prime_tokens])
            if self.use_cuda:
                src_tensor = src_tensor.cuda()
            
            # Encode
            src_emb = self.embedding(src_tensor) * math.sqrt(self.d_model)
            src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
            memory = self.encoder(src_emb)
            
            # Decode autoregressively
            generated = prime_str
            tgt_tokens = [data.char2idx['<']]  # Start with BOS token
            
            for _ in range(max_len):
                tgt_tensor = torch.LongTensor([tgt_tokens])
                if self.use_cuda:
                    tgt_tensor = tgt_tensor.cuda()
                
                tgt_mask = self._create_causal_mask(len(tgt_tokens))
                
                tgt_emb = self.embedding(tgt_tensor) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
                
                output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                output = self.output_projection(output)
                
                # Sample next token
                logits = output[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                next_char = data.all_characters[next_token]
                generated += next_char
                tgt_tokens.append(next_token)
                
                if next_char == end_token:
                    break
            
        self.train()
        return generated
    
    def train_step(self, inp, target):
        """Single training step for encoder-decoder."""
        self.optimizer.zero_grad()
        
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # Use input as source and target as decoder input
        output = self.forward(inp, inp)
        loss = self.criterion(output.view(-1, self.vocab_size), target.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def fit(self, data, n_iterations, all_losses=[], print_every=100,
            plot_every=10, augment=False):
        """Train the encoder-decoder model."""
        start = time.time()
        loss_avg = 0
        
        if augment:
            smiles_augmentation = SmilesEnumerator()
        else:
            smiles_augmentation = None
        
        for epoch in trange(1, n_iterations + 1, desc='Training BART-style Transformer...'):
            inp, target = data.random_training_set(smiles_augmentation)
            loss = self.train_step(inp, target)
            loss_avg += loss
            
            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,
                                              epoch / n_iterations * 100, loss))
                print(self.generate(data=data, prime_str='<', max_len=100), '\n')
            
            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
        
        return all_losses
    
    def evaluate(self, data, prime_str='<', end_token='>', predict_len=100):
        """Generate sequence (compatible with original interface)."""
        return self.generate(data, prime_str, end_token, predict_len)
    
    def load_model(self, path):
        """Load model weights."""
        weights = torch.load(path, map_location='cpu' if not self.use_cuda else None)
        self.load_state_dict(weights)
    
    def save_model(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def change_lr(self, new_lr):
        """Update learning rate."""
        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr


# Factory function for easy model creation
def create_transformer_generator(model_type='gpt', vocab_size=None, **kwargs):
    """
    Factory function to create transformer generators.
    
    Parameters
    ----------
    model_type: str
        Type of transformer ('gpt' or 'bart')
    vocab_size: int
        Vocabulary size
    **kwargs: dict
        Additional model parameters
    
    Returns
    -------
    model: nn.Module
        Transformer generator model
    """
    if model_type.lower() == 'gpt':
        return GPTStyleGenerator(vocab_size, **kwargs)
    elif model_type.lower() == 'bart':
        return BARTStyleGenerator(vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'gpt' or 'bart'.")
