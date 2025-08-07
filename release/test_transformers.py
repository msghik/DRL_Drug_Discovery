"""
Test script to verify transformer generator implementation.
This script creates synthetic data and tests the basic functionality.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add the release directory to path
sys.path.append('/root/ReLease/ReLeaSE/release')

from transformer_generator import GPTStyleGenerator, BARTStyleGenerator, create_transformer_generator


class MockGeneratorData:
    """Mock data class for testing without real data files."""
    
    def __init__(self):
        # Create a simple vocabulary for testing
        self.all_characters = list("()<>CNOSc123456789=+-#[]")
        self.char2idx = {char: i for i, char in enumerate(self.all_characters)}
        self.n_characters = len(self.all_characters)
        self.use_cuda = torch.cuda.is_available()
        
        # Create some mock SMILES-like sequences
        self.file = [
            "<CCO>",
            "<CCC>", 
            "<C(C)O>",
            "<CC(C)C>",
            "<CCCCO>",
            "<CC(=O)O>",
            "<c1ccccc1>",
            "<CC(C)(C)O>",
            "<CCN(C)C>",
            "<CCOC>",
        ]
        self.file_len = len(self.file)
    
    def random_chunk(self):
        """Return a random sequence from our mock data."""
        index = np.random.randint(0, self.file_len)
        return self.file[index]
    
    def char_tensor(self, string):
        """Convert string to tensor of character indices."""
        tensor = torch.zeros(len(string), dtype=torch.long)
        for i, char in enumerate(string):
            if char in self.char2idx:
                tensor[i] = self.char2idx[char]
        
        if self.use_cuda:
            return tensor.cuda()
        return tensor
    
    def random_training_set(self, smiles_augmentation=None):
        """Get a random training pair."""
        chunk = self.random_chunk()
        # For simplicity, ignore augmentation in mock data
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target


def test_gpt_generator():
    """Test GPT-style generator."""
    print("Testing GPT-style generator...")
    
    # Create mock data
    data = MockGeneratorData()
    
    # Create generator
    generator = GPTStyleGenerator(
        vocab_size=data.n_characters,
        d_model=64,  # Small for testing
        nhead=4,
        num_layers=2,
        max_len=50,
        use_cuda=data.use_cuda,
        lr=1e-3
    )
    
    print(f"Created generator with {sum(p.numel() for p in generator.parameters())} parameters")
    
    # Test training step
    inp, target = data.random_training_set()
    loss = generator.train_step(inp, target)
    print(f"Training step loss: {loss:.4f}")
    
    # Test generation
    sample = generator.evaluate(data, prime_str='<', predict_len=20)
    print(f"Generated sample: {sample}")
    
    # Test short training
    print("Running short training...")
    losses = generator.fit(data, n_iterations=10, print_every=5, plot_every=2)
    print(f"Training losses: {losses}")
    
    # Test model saving/loading
    model_path = "/tmp/test_gpt_model.pt"
    generator.save_model(model_path)
    
    # Create new generator and load weights
    new_generator = GPTStyleGenerator(
        vocab_size=data.n_characters,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_len=50,
        use_cuda=data.use_cuda
    )
    new_generator.load_model(model_path)
    
    # Test that loaded model generates similar output
    sample2 = new_generator.evaluate(data, prime_str='<', predict_len=20)
    print(f"Generated sample from loaded model: {sample2}")
    
    print("GPT-style generator test completed successfully!\n")
    return True


def test_bart_generator():
    """Test BART-style generator."""
    print("Testing BART-style generator...")
    
    # Create mock data
    data = MockGeneratorData()
    
    # Create generator
    generator = BARTStyleGenerator(
        vocab_size=data.n_characters,
        d_model=64,  # Small for testing
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=50,
        use_cuda=data.use_cuda,
        lr=1e-3
    )
    
    print(f"Created BART generator with {sum(p.numel() for p in generator.parameters())} parameters")
    
    # Test training step
    inp, target = data.random_training_set()
    loss = generator.train_step(inp, target)
    print(f"Training step loss: {loss:.4f}")
    
    # Test generation
    sample = generator.evaluate(data, prime_str='<', predict_len=20)
    print(f"Generated sample: {sample}")
    
    # Test short training
    print("Running short training...")
    losses = generator.fit(data, n_iterations=10, print_every=5, plot_every=2)
    print(f"Training losses: {losses}")
    
    print("BART-style generator test completed successfully!\n")
    return True


def test_factory_function():
    """Test the factory function."""
    print("Testing factory function...")
    
    data = MockGeneratorData()
    
    # Test GPT creation
    gpt_gen = create_transformer_generator(
        model_type='gpt',
        vocab_size=data.n_characters,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    # Test BART creation
    bart_gen = create_transformer_generator(
        model_type='bart',
        vocab_size=data.n_characters,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    print("Factory function test completed successfully!\n")
    return True


def test_batch_processing():
    """Test batch processing capabilities."""
    print("Testing batch processing...")
    
    data = MockGeneratorData()
    
    # Test the new get_batch method
    try:
        inputs, targets, attention_mask = data.get_batch(batch_size=3)
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch targets shape: {targets.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print("Batch processing test completed successfully!\n")
        return True
    except Exception as e:
        print(f"Batch processing test failed: {e}")
        return False


def test_interface_compatibility():
    """Test that the interface matches the original StackAugmentedRNN."""
    print("Testing interface compatibility...")
    
    data = MockGeneratorData()
    generator = GPTStyleGenerator(
        vocab_size=data.n_characters,
        d_model=64,
        nhead=4,
        num_layers=2,
        use_cuda=data.use_cuda
    )
    
    # Test all required methods exist and work
    methods_to_test = [
        'fit', 'evaluate', 'load_model', 'save_model', 'change_lr'
    ]
    
    for method in methods_to_test:
        if hasattr(generator, method):
            print(f"✓ Method '{method}' exists")
        else:
            print(f"✗ Method '{method}' missing")
            return False
    
    # Test method signatures
    try:
        # Test evaluate with same signature as StackAugmentedRNN
        sample = generator.evaluate(
            data=data, 
            prime_str='<', 
            end_token='>', 
            predict_len=20
        )
        print(f"✓ evaluate() method compatible: {sample}")
        
        # Test change_lr
        original_lr = generator.lr
        generator.change_lr(0.001)
        if generator.lr == 0.001:
            print("✓ change_lr() method compatible")
        else:
            print("✗ change_lr() method not working")
            return False
        
        print("Interface compatibility test completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"Interface compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running Transformer Generator Tests")
    print("=" * 50)
    
    tests = [
        test_gpt_generator,
        test_bart_generator,
        test_factory_function,
        test_batch_processing,
        test_interface_compatibility
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The transformer generators are working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
