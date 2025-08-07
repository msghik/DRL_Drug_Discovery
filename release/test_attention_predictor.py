"""
Test script for the Attention-Augmented LSTM Predictor.

This script tests the core functionality without requiring external dependencies.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock

# Add the release directory to path
sys.path.append('/root/ReLease/ReLeaSE/release')

try:
    from attention_predictor import (
        SelfAttention, 
        AttentionPooling, 
        AttentionAugmentedLSTMPredictor,
        AttentionAugmentedPredictor
    )
    print("✓ Successfully imported attention predictor modules")
except ImportError as e:
    print(f"✗ Failed to import attention predictor modules: {e}")
    sys.exit(1)


class MockData:
    """Mock data class for testing."""
    
    def __init__(self):
        # Simple vocabulary for testing
        self.tokens = 'CNOSPH()=123456789'
        self.vocab_size = len(self.tokens)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.tokens)}
        
        # Simple test SMILES
        self.test_smiles = [
            'CCO',           # Ethanol
            'c1ccccc1',      # Benzene
            'CCCCCC',        # Hexane
            'CC(C)O',        # Isopropanol
            'CN(C)C',        # Trimethylamine
        ]
        
        # Mock property values (LogP-like)
        self.test_properties = [-0.31, 2.13, 3.90, 0.05, 0.16]
    
    def tokenize_smiles(self, smiles_list, max_len=20):
        """Convert SMILES to token sequences."""
        sequences = []
        lengths = []
        
        for smiles in smiles_list:
            sequence = [self.char_to_idx.get(char, 0) for char in smiles[:max_len]]
            sequences.append(sequence)
            lengths.append(len(sequence))
        
        # Pad sequences
        max_seq_len = max(lengths)
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [0] * (max_seq_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return torch.LongTensor(padded_sequences), torch.LongTensor(lengths)


def test_self_attention():
    """Test the SelfAttention module."""
    print("\nTesting SelfAttention module...")
    
    try:
        # Create test data
        batch_size, seq_len, hidden_size = 2, 10, 64
        x = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)
        
        # Create attention layer
        attention = SelfAttention(hidden_size, num_heads=8)
        
        # Forward pass
        output, attention_weights = attention(x, mask)
        
        # Check output shapes
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len), \
            f"Expected {(batch_size, 8, seq_len, seq_len)}, got {attention_weights.shape}"
        
        print("✓ SelfAttention module test passed")
        return True
        
    except Exception as e:
        print(f"✗ SelfAttention module test failed: {e}")
        return False


def test_attention_pooling():
    """Test the AttentionPooling module."""
    print("\nTesting AttentionPooling module...")
    
    try:
        # Create test data
        batch_size, seq_len, hidden_size = 2, 10, 64
        x = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)
        
        # Create pooling layer
        pooling = AttentionPooling(hidden_size)
        
        # Forward pass
        output, attention_weights = pooling(x, mask)
        
        # Check output shapes
        assert output.shape == (batch_size, hidden_size), \
            f"Expected {(batch_size, hidden_size)}, got {output.shape}"
        assert attention_weights.shape == (batch_size, seq_len), \
            f"Expected {(batch_size, seq_len)}, got {attention_weights.shape}"
        
        # Check attention weights sum to 1
        weights_sum = attention_weights.sum(dim=1)
        assert torch.allclose(weights_sum, torch.ones(batch_size), atol=1e-6), \
            "Attention weights should sum to 1"
        
        print("✓ AttentionPooling module test passed")
        return True
        
    except Exception as e:
        print(f"✗ AttentionPooling module test failed: {e}")
        return False


def test_lstm_predictor():
    """Test the AttentionAugmentedLSTMPredictor module."""
    print("\nTesting AttentionAugmentedLSTMPredictor module...")
    
    try:
        # Create test data
        mock_data = MockData()
        vocab_size = mock_data.vocab_size
        
        # Create model
        model = AttentionAugmentedLSTMPredictor(
            vocab_size=vocab_size,
            embedding_dim=32,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            num_attention_heads=4,
            num_dense_layers=1,
            dense_hidden_size=32,
            output_size=1,
            dropout=0.1
        )
        
        # Create test input
        sequences, lengths = mock_data.tokenize_smiles(mock_data.test_smiles)
        batch_size = sequences.size(0)
        
        # Forward pass
        predictions = model(sequences, lengths)
        
        # Check output shape
        assert predictions.shape == (batch_size, 1), \
            f"Expected {(batch_size, 1)}, got {predictions.shape}"
        
        # Test with attention return
        predictions, attention_info = model(sequences, lengths, return_attention=True)
        
        # Check attention info
        assert 'self_attention_weights' in attention_info
        assert 'pooling_attention_weights' in attention_info
        
        print("✓ AttentionAugmentedLSTMPredictor module test passed")
        return True
        
    except Exception as e:
        print(f"✗ AttentionAugmentedLSTMPredictor module test failed: {e}")
        return False


def test_attention_predictor():
    """Test the AttentionAugmentedPredictor wrapper."""
    print("\nTesting AttentionAugmentedPredictor wrapper...")
    
    try:
        # Create test data
        mock_data = MockData()
        
        # Create predictor
        predictor = AttentionAugmentedPredictor(
            vocab_size=mock_data.vocab_size,
            model_params={
                'embedding_dim': 32,
                'lstm_hidden_size': 64,
                'num_lstm_layers': 1,
                'num_attention_heads': 4,
                'num_dense_layers': 1,
                'dense_hidden_size': 32,
                'output_size': 1,
                'dropout': 0.1
            },
            ensemble_size=2,  # Small ensemble for testing
            use_cuda=False
        )
        
        # Test prediction
        canonical_smiles, predictions, invalid_smiles = predictor.predict(
            mock_data.test_smiles, mock_data.tokens
        )
        
        # Check results
        assert len(canonical_smiles) > 0, "Should have some valid predictions"
        assert len(predictions) == len(canonical_smiles), \
            "Number of predictions should match number of valid SMILES"
        assert predictions.shape[1] == 1, "Should have 1 output dimension"
        
        # Test with attention return
        canonical_smiles, predictions, invalid_smiles, attention_info = predictor.predict(
            mock_data.test_smiles, mock_data.tokens, return_attention=True
        )
        
        # Check attention info
        assert 'self_attention_weights' in attention_info
        assert 'pooling_attention_weights' in attention_info
        
        print("✓ AttentionAugmentedPredictor wrapper test passed")
        return True
        
    except Exception as e:
        print(f"✗ AttentionAugmentedPredictor wrapper test failed: {e}")
        return False


def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    try:
        # Create test data
        mock_data = MockData()
        
        # Create model
        model = AttentionAugmentedLSTMPredictor(
            vocab_size=mock_data.vocab_size,
            embedding_dim=32,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            num_attention_heads=4,
            num_dense_layers=1,
            dense_hidden_size=32,
            output_size=1,
            dropout=0.1
        )
        
        # Create test input
        sequences, lengths = mock_data.tokenize_smiles(mock_data.test_smiles)
        targets = torch.FloatTensor(mock_data.test_properties).unsqueeze(1)
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        predictions = model(sequences, lengths)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        # Check that loss is a valid number
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"
        
        print(f"✓ Training step test passed (loss: {loss.item():.4f})")
        return True
        
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        return False


def test_attention_weights_properties():
    """Test properties of attention weights."""
    print("\nTesting attention weights properties...")
    
    try:
        # Create test data
        mock_data = MockData()
        
        # Create model
        model = AttentionAugmentedLSTMPredictor(
            vocab_size=mock_data.vocab_size,
            embedding_dim=32,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            num_attention_heads=4,
            num_dense_layers=1,
            dense_hidden_size=32,
            output_size=1,
            dropout=0.1
        )
        
        # Create test input
        sequences, lengths = mock_data.tokenize_smiles(['CCO'])  # Single molecule
        
        # Get attention weights
        model.eval()
        with torch.no_grad():
            predictions, attention_info = model(sequences, lengths, return_attention=True)
        
        # Check pooling attention weights
        pooling_weights = attention_info['pooling_attention_weights'][0]  # First sample
        
        # Should sum to approximately 1
        weights_sum = pooling_weights.sum()
        assert abs(weights_sum - 1.0) < 1e-5, f"Pooling weights should sum to 1, got {weights_sum}"
        
        # Should be non-negative
        assert (pooling_weights >= 0).all(), "Pooling weights should be non-negative"
        
        # Check self-attention weights
        self_attention_weights = attention_info['self_attention_weights'][0]  # First sample
        
        # Should sum to 1 along the last dimension
        self_weights_sum = self_attention_weights.sum(dim=-1)
        expected_shape = self_attention_weights.shape[:-1]
        assert torch.allclose(self_weights_sum, torch.ones(expected_shape), atol=1e-5), \
            "Self-attention weights should sum to 1 along last dimension"
        
        print("✓ Attention weights properties test passed")
        return True
        
    except Exception as e:
        print(f"✗ Attention weights properties test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("Running Attention-Augmented LSTM Predictor Tests")
    print("=" * 60)
    
    tests = [
        test_self_attention,
        test_attention_pooling,
        test_lstm_predictor,
        test_attention_predictor,
        test_training_step,
        test_attention_weights_properties,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The attention-augmented predictor is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
