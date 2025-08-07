"""
Example script demonstrating how to replace StackAugmentedRNN with transformer generators.
This script shows both training and integration with the ReLeaSE reinforcement learning framework.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the release directory to path
sys.path.append('/root/ReLease/ReLeaSE/release')

from data import GeneratorData, PredictorData
from transformer_generator import GPTStyleGenerator, BARTStyleGenerator, create_transformer_generator
from reinforcement import Reinforcement
from predictor import VanillaQSAR
from utils import get_fp


def example_reward_function(smiles, predictor, threshold=0.5):
    """
    Example reward function for reinforcement learning.
    
    Parameters
    ----------
    smiles: str
        SMILES string to evaluate
    predictor: object
        Predictive model
    threshold: float
        Threshold for reward calculation
    
    Returns
    -------
    reward: float
        Calculated reward
    """
    try:
        # Simple example: reward based on length and predicted property
        length_penalty = max(0, 1 - len(smiles) / 100)  # Prefer shorter molecules
        
        # Get prediction (you would use your actual predictor here)
        processed_objects, prediction, invalid_objects = predictor.predict([smiles], get_features=get_fp)
        
        if len(invalid_objects) > 0:
            return 0.0  # Invalid molecule
        
        # Combine prediction with length penalty
        if len(prediction) > 0:
            property_reward = 1.0 if prediction[0] > threshold else 0.0
            return property_reward * length_penalty
        else:
            return 0.0
    except:
        return 0.0


def train_transformer_generator(data_path, model_type='gpt', n_iterations=1000):
    """
    Train a transformer generator on molecular data.
    
    Parameters
    ----------
    data_path: str
        Path to training data file
    model_type: str
        Type of transformer ('gpt' or 'bart')
    n_iterations: int
        Number of training iterations
    
    Returns
    -------
    generator: nn.Module
        Trained generator model
    data: GeneratorData
        Data object
    """
    print(f"Training {model_type.upper()}-style transformer generator...")
    
    # Load data
    data = GeneratorData(
        training_data_path=data_path,
        delimiter='\t',  # Assuming tab-separated file
        cols_to_read=[0],  # First column contains SMILES
        max_len=120,
        use_cuda=torch.cuda.is_available()
    )
    
    print(f"Loaded {data.file_len} molecules")
    print(f"Vocabulary size: {data.n_characters}")
    print(f"Vocabulary: {data.all_characters}")
    
    # Create transformer generator
    generator = create_transformer_generator(
        model_type=model_type,
        vocab_size=data.n_characters,
        d_model=512,
        nhead=8,
        num_layers=6,
        max_len=150,
        dropout=0.1,
        use_cuda=data.use_cuda,
        lr=1e-4
    )
    
    print(f"Created {model_type.upper()}-style generator with {sum(p.numel() for p in generator.parameters())} parameters")
    
    # Train the generator
    losses = generator.fit(
        data=data,
        n_iterations=n_iterations,
        print_every=100,
        plot_every=10,
        augment=True  # Use SMILES augmentation
    )
    
    print("Training completed!")
    
    # Generate some examples
    print("\nGenerating sample molecules:")
    for i in range(5):
        sample = generator.evaluate(data, prime_str='<', predict_len=100)
        print(f"Sample {i+1}: {sample}")
    
    return generator, data, losses


def train_with_reinforcement_learning(generator, data, predictor_data_path):
    """
    Demonstrate reinforcement learning with the transformer generator.
    
    Parameters
    ----------
    generator: nn.Module
        Pretrained generator
    data: GeneratorData
        Generator data object
    predictor_data_path: str
        Path to predictor training data
    
    Returns
    -------
    rl_agent: Reinforcement
        Trained reinforcement learning agent
    """
    print("Setting up reinforcement learning...")
    
    # Create a simple predictor (you can replace this with more sophisticated models)
    # For this example, we'll create a dummy predictor
    from sklearn.ensemble import RandomForestClassifier
    
    # Load predictor data
    predictor_data = PredictorData(
        path=predictor_data_path,
        delimiter=',',
        cols=[0, 1],  # SMILES and target property
        get_features=get_fp,  # Use molecular fingerprints
        has_label=True
    )
    
    # Create and train predictor
    predictor = VanillaQSAR(
        model_instance=RandomForestClassifier,
        model_params={'n_estimators': 100, 'random_state': 42},
        model_type='classifier',
        ensemble_size=5
    )
    
    metrics = predictor.fit_model(predictor_data)
    print(f"Predictor trained with metrics: {metrics}")
    
    # Create reinforcement learning agent
    rl_agent = Reinforcement(
        generator=generator,
        predictor=predictor,
        get_reward=example_reward_function
    )
    
    # Run reinforcement learning
    print("Starting reinforcement learning...")
    total_rewards = []
    rl_losses = []
    
    for epoch in range(50):  # 50 RL epochs
        reward, loss = rl_agent.policy_gradient(
            data=data,
            n_batch=10,
            gamma=0.97,
            std_smiles=True,  # Standardize SMILES
            grad_clipping=5.0,
            threshold=0.5  # Threshold for reward function
        )
        
        total_rewards.append(reward)
        rl_losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Reward = {reward:.4f}, Loss = {loss:.4f}")
            
            # Generate some samples to see improvement
            print("Generated samples:")
            for i in range(3):
                sample = generator.evaluate(data, prime_str='<', predict_len=100)
                print(f"  {sample}")
    
    print("Reinforcement learning completed!")
    return rl_agent, total_rewards, rl_losses


def compare_architectures(data_path, n_iterations=500):
    """
    Compare different transformer architectures.
    
    Parameters
    ----------
    data_path: str
        Path to training data
    n_iterations: int
        Number of training iterations
    """
    print("Comparing transformer architectures...")
    
    results = {}
    
    for model_type in ['gpt', 'bart']:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}-style model...")
        print(f"{'='*50}")
        
        generator, data, losses = train_transformer_generator(
            data_path=data_path,
            model_type=model_type,
            n_iterations=n_iterations
        )
        
        results[model_type] = {
            'generator': generator,
            'data': data,
            'losses': losses,
            'final_loss': losses[-1] if losses else float('inf')
        }
        
        # Save the model
        model_path = f"/root/ReLease/ReLeaSE/checkpoints/{model_type}_transformer.pt"
        generator.save_model(model_path)
        print(f"Saved {model_type} model to {model_path}")
    
    # Compare results
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    
    for model_type, result in results.items():
        print(f"{model_type.upper()}-style:")
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"  Parameters: {sum(p.numel() for p in result['generator'].parameters())}")
        
        # Generate examples
        print("  Sample generations:")
        for i in range(3):
            sample = result['generator'].evaluate(result['data'], prime_str='<', predict_len=100)
            print(f"    {sample}")
        print()
    
    return results


def main():
    """Main example function."""
    print("ReLeaSE Transformer Generator Examples")
    print("="*50)
    
    # Paths (adjust these to your actual data files)
    training_data_path = "/root/ReLease/ReLeaSE/data/chembl_22_clean_1576904_sorted_std_final.smi"
    predictor_data_path = "/root/ReLease/ReLeaSE/data/logP_labels.csv"
    
    # Check if data files exist
    if not Path(training_data_path).exists():
        print(f"Warning: Training data not found at {training_data_path}")
        print("Please adjust the path in the script or provide your own data.")
        return
    
    try:
        # Example 1: Train a single GPT-style generator
        print("\n1. Training GPT-style transformer...")
        generator, data, losses = train_transformer_generator(
            data_path=training_data_path,
            model_type='gpt',
            n_iterations=100  # Reduced for demo
        )
        
        # Example 2: Use with reinforcement learning (if predictor data exists)
        if Path(predictor_data_path).exists():
            print("\n2. Demonstrating reinforcement learning...")
            rl_agent, rewards, rl_losses = train_with_reinforcement_learning(
                generator, data, predictor_data_path
            )
        else:
            print(f"\n2. Skipping RL demo - predictor data not found at {predictor_data_path}")
        
        # Example 3: Compare architectures (reduced iterations for demo)
        print("\n3. Comparing architectures...")
        results = compare_architectures(training_data_path, n_iterations=100)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing data files or dependencies.")
        print("Please ensure you have the required data files and all dependencies installed.")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()
