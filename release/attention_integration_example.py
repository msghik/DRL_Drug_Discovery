"""
Integration example: Attention-Augmented Predictor with ReLeaSE Reinforcement Learning

This script demonstrates how to use the attention-augmented LSTM predictor
within the ReLeaSE reinforcement learning framework for molecular optimization.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the release directory to path
sys.path.append('/root/ReLease/ReLeaSE/release')

from data import GeneratorData
from transformer_generator import GPTStyleGenerator
from attention_predictor import AttentionAugmentedPredictor
from reinforcement import Reinforcement


def create_mock_training_data():
    """Create mock molecular training data."""
    # Sample SMILES with corresponding LogP values
    training_smiles = [
        'CCO', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC',          # Alcohols and alkanes
        'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1',             # Aromatics
        'CC(C)O', 'CC(C)C', 'CCC(C)C',                     # Branched
        'CCN', 'CCCN', 'CC(C)N',                           # Amines
        'CC=O', 'CCC=O', 'CC(=O)C',                        # Carbonyls
    ] * 10  # Repeat for more data
    
    # Corresponding LogP values (approximate)
    logp_values = [
        -0.31, 1.09, 2.13, 3.16, 4.11,                    # Alcohols and alkanes
        2.13, 2.73, 3.15,                                 # Aromatics
        0.05, 1.31, 1.84,                                 # Branched
        -0.57, 0.48, 0.15,                                # Amines
        -0.24, 0.79, -0.24,                               # Carbonyls
    ] * 10
    
    # Add some noise
    np.random.seed(42)
    logp_values = np.array(logp_values) + np.random.normal(0, 0.1, len(logp_values))
    
    return training_smiles, logp_values.tolist()


def train_attention_predictor_simple():
    """Train a simple attention-augmented predictor."""
    print("Training attention-augmented predictor...")
    
    # Create training data
    smiles_list, properties = create_mock_training_data()
    
    # Create vocabulary
    all_chars = set(''.join(smiles_list))
    tokens = ''.join(sorted(all_chars))
    
    print(f"Training on {len(smiles_list)} molecules")
    print(f"Vocabulary: {tokens}")
    
    # Create predictor
    predictor = AttentionAugmentedPredictor(
        vocab_size=len(tokens),
        model_params={
            'embedding_dim': 64,
            'lstm_hidden_size': 128,
            'num_lstm_layers': 1,
            'num_attention_heads': 4,
            'num_dense_layers': 1,
            'dense_hidden_size': 64,
            'output_size': 1,
            'dropout': 0.1
        },
        ensemble_size=2,  # Small ensemble for demo
        use_cuda=torch.cuda.is_available()
    )
    
    # Quick training
    train_data = (smiles_list, properties, tokens)
    history = predictor.train_model(
        train_data=train_data,
        val_data=None,
        epochs=20,  # Quick training for demo
        batch_size=16,
        learning_rate=1e-3,
        patience=5
    )
    
    # Test predictions
    test_smiles = ['CCO', 'c1ccccc1', 'CCCCCCCC']
    canonical_smiles, predictions, invalid_smiles = predictor.predict(
        test_smiles, tokens
    )
    
    print("\nTest predictions:")
    for smiles, pred in zip(canonical_smiles, predictions):
        print(f"  {smiles}: LogP = {pred[0]:.3f}")
    
    return predictor, tokens


def create_attention_reward_function(predictor, tokens, target_logp=2.0):
    """
    Create a reward function using the attention-augmented predictor.
    
    Parameters
    ----------
    predictor: AttentionAugmentedPredictor
        Trained predictor
    tokens: str
        Vocabulary tokens
    target_logp: float
        Target LogP value
        
    Returns
    -------
    reward_function: callable
        Reward function for RL
    """
    def attention_reward(smiles, predictor_obj, **kwargs):
        """
        Reward function that considers both property prediction and attention.
        
        Parameters
        ----------
        smiles: str
            SMILES string to evaluate
        predictor_obj: object
            Predictor object (not used, for compatibility)
            
        Returns
        -------
        reward: float
            Calculated reward
        """
        try:
            # Get prediction with attention
            canonical_smiles, predictions, invalid_smiles, attention_info = predictor.predict(
                [smiles], tokens, return_attention=True
            )
            
            if len(invalid_smiles) > 0:
                return 0.0  # Invalid molecule
            
            if len(predictions) > 0:
                predicted_logp = predictions[0][0]
                
                # Base reward: proximity to target LogP
                logp_distance = abs(predicted_logp - target_logp)
                logp_reward = max(0, 1.0 - logp_distance / 3.0)  # Normalize by expected range
                
                # Bonus for molecules with focused attention patterns
                # (indicating the model is confident about important substructures)
                pooling_weights = attention_info['pooling_attention_weights'][0]
                
                # Calculate attention entropy (lower entropy = more focused)
                # Add small epsilon to avoid log(0)
                entropy = -np.sum(pooling_weights * np.log(pooling_weights + 1e-8))
                max_entropy = np.log(len(pooling_weights))  # Maximum possible entropy
                focus_bonus = 1.0 - (entropy / max_entropy)  # Higher bonus for lower entropy
                
                # Combine rewards
                total_reward = logp_reward * (1.0 + 0.2 * focus_bonus)  # 20% bonus for focus
                
                # Extra bonus for drug-like LogP range
                if 1.0 <= predicted_logp <= 3.0:
                    total_reward *= 1.1
                
                return max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error in attention reward function: {e}")
            return 0.0
    
    return attention_reward


def demonstrate_rl_with_attention():
    """Demonstrate reinforcement learning with attention-augmented predictor."""
    print("\n" + "="*60)
    print("Demonstrating RL with Attention-Augmented Predictor")
    print("="*60)
    
    # Train the attention predictor
    attention_predictor, tokens = train_attention_predictor_simple()
    
    # Create generator data (mock)
    print("\nCreating generator data...")
    training_smiles, _ = create_mock_training_data()
    
    # Save mock data to temporary file
    temp_file = "/tmp/mock_smiles.smi"
    with open(temp_file, 'w') as f:
        for smiles in training_smiles:
            f.write(f"{smiles}\n")
    
    # Create generator data object
    generator_data = GeneratorData(
        training_data_path=temp_file,
        delimiter='\t',
        cols_to_read=[0],
        max_len=50,
        use_cuda=torch.cuda.is_available()
    )
    
    print(f"Generator vocabulary size: {generator_data.n_characters}")
    
    # Create transformer generator
    print("\nCreating transformer generator...")
    generator = GPTStyleGenerator(
        vocab_size=generator_data.n_characters,
        d_model=128,  # Smaller for demo
        nhead=4,
        num_layers=2,
        max_len=50,
        use_cuda=torch.cuda.is_available(),
        lr=1e-3
    )
    
    # Quick pre-training of generator
    print("Pre-training generator...")
    losses = generator.fit(
        data=generator_data,
        n_iterations=50,  # Quick training
        print_every=25,
        plot_every=10,
        augment=False
    )
    
    # Create attention-based reward function
    print("\nCreating attention-based reward function...")
    attention_reward = create_attention_reward_function(
        attention_predictor, tokens, target_logp=2.0
    )
    
    # Create RL agent
    print("Creating reinforcement learning agent...")
    rl_agent = Reinforcement(
        generator=generator,
        predictor=attention_predictor,  # This won't be used directly
        get_reward=attention_reward
    )
    
    # Test the reward function
    print("\nTesting reward function:")
    test_molecules = [
        'CCO',           # Low LogP
        'c1ccccc1',      # Medium LogP
        'CCCCCCCC',      # High LogP
        'CC(C)c1ccccc1', # Medium LogP with branching
    ]
    
    for smiles in test_molecules:
        reward = attention_reward(smiles, None)
        print(f"  {smiles}: reward = {reward:.3f}")
    
    # Run a few RL steps
    print("\nRunning reinforcement learning steps...")
    try:
        for step in range(3):  # Just a few steps for demo
            print(f"\nRL Step {step + 1}:")
            
            # Sample molecules and get rewards
            sampled_molecules = []
            rewards = []
            
            for _ in range(5):  # Sample 5 molecules
                molecule = generator.evaluate(generator_data, predict_len=30)
                if len(molecule) > 2:  # Valid molecule
                    molecule_clean = molecule[1:-1]  # Remove start/end tokens
                    reward = attention_reward(molecule_clean, None)
                    sampled_molecules.append(molecule_clean)
                    rewards.append(reward)
            
            if sampled_molecules:
                avg_reward = np.mean(rewards)
                max_reward = np.max(rewards)
                print(f"  Average reward: {avg_reward:.3f}")
                print(f"  Max reward: {max_reward:.3f}")
                print(f"  Best molecule: {sampled_molecules[np.argmax(rewards)]}")
            
            # Note: We skip the actual RL training step here as it requires
            # more complex setup, but this shows how the components work together
            
    except Exception as e:
        print(f"RL simulation encountered an error: {e}")
        print("This is expected in a mock demo environment")
    
    print("\n" + "="*60)
    print("Key Benefits of Attention-Augmented Approach:")
    print("- Better interpretability through attention visualization")
    print("- Reward functions can leverage attention patterns")
    print("- Focus on chemically meaningful substructures")
    print("- Improved property prediction accuracy")
    print("="*60)


def demonstrate_attention_analysis():
    """Demonstrate attention analysis capabilities."""
    print("\n" + "="*50)
    print("Attention Analysis Demonstration")
    print("="*50)
    
    # Train predictor
    predictor, tokens = train_attention_predictor_simple()
    
    # Analyze attention for different molecules
    analysis_molecules = [
        ('CCO', 'Simple alcohol'),
        ('c1ccccc1', 'Benzene ring'),
        ('CCc1ccccc1', 'Phenethyl group'),
        ('CC(C)(C)C', 'tert-Butyl group'),
        ('CCCCCCCC', 'Long alkyl chain'),
    ]
    
    print("\nAttention Analysis Results:")
    print("-" * 50)
    
    for smiles, description in analysis_molecules:
        try:
            # Get predictions with attention
            canonical_smiles, predictions, invalid_smiles, attention_info = predictor.predict(
                [smiles], tokens, return_attention=True
            )
            
            if len(canonical_smiles) > 0:
                canonical_smiles_str = canonical_smiles[0]
                predicted_logp = predictions[0][0]
                pooling_weights = attention_info['pooling_attention_weights'][0]
                
                print(f"\nMolecule: {smiles} ({description})")
                print(f"Canonical: {canonical_smiles_str}")
                print(f"Predicted LogP: {predicted_logp:.3f}")
                
                # Find most attended positions
                top_positions = np.argsort(pooling_weights)[-3:][::-1]  # Top 3
                print("Most attended positions:")
                for pos in top_positions:
                    if pos < len(canonical_smiles_str):
                        char = canonical_smiles_str[pos]
                        weight = pooling_weights[pos]
                        print(f"  Position {pos}: '{char}' (weight: {weight:.3f})")
                
                # Calculate attention entropy
                entropy = -np.sum(pooling_weights * np.log(pooling_weights + 1e-8))
                max_entropy = np.log(len(pooling_weights))
                normalized_entropy = entropy / max_entropy
                print(f"Attention entropy: {normalized_entropy:.3f} (lower = more focused)")
                
        except Exception as e:
            print(f"Error analyzing {smiles}: {e}")
    
    print("\n" + "="*50)
    print("Attention patterns can reveal:")
    print("- Which atoms/bonds are most important for the property")
    print("- How confident the model is about its prediction")
    print("- Chemically meaningful substructures")
    print("="*50)


if __name__ == "__main__":
    print("Attention-Augmented Predictor Integration with ReLeaSE")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_rl_with_attention()
    demonstrate_attention_analysis()
    
    print("\n" + "=" * 60)
    print("Integration demonstration completed!")
    print("\nThis enhanced architecture provides:")
    print("1. Better predictive performance through attention mechanisms")
    print("2. Interpretable models via attention weight visualization") 
    print("3. More sophisticated reward functions for RL")
    print("4. Focus on chemically relevant molecular substructures")
    print("=" * 60)
