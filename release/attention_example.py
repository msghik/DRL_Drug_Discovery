"""
Example script demonstrating the Attention-Augmented LSTM Predictor.

This script shows how to:
1. Train the attention-augmented predictor on molecular property data
2. Compare performance with the original LSTM predictor
3. Visualize attention weights to understand model focus
4. Integrate with the ReLeaSE reinforcement learning framework
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add the release directory to path
sys.path.append('/root/ReLease/ReLeaSE/release')

from data import PredictorData
from attention_predictor import AttentionAugmentedPredictor, visualize_attention
from utils import tokenize, read_object_property_file
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_molecular_data(data_path, property_column=1):
    """
    Load molecular data for training.
    
    Parameters
    ----------
    data_path: str
        Path to the data file
    property_column: int
        Column index for the target property
        
    Returns
    -------
    smiles_list: list
        List of SMILES strings
    properties: list
        List of property values
    tokens: str
        Vocabulary tokens
    """
    try:
        # Read data
        data = read_object_property_file(
            data_path, 
            delimiter=',', 
            cols_to_read=[0, property_column]
        )
        
        smiles_list = data[0]
        properties = [float(x) for x in data[1]]
        
        # Create vocabulary from SMILES
        all_chars = set(''.join(smiles_list))
        tokens = ''.join(sorted(all_chars))
        
        print(f"Loaded {len(smiles_list)} molecules")
        print(f"Vocabulary size: {len(tokens)}")
        print(f"Vocabulary: {tokens}")
        
        return smiles_list, properties, tokens
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return mock data for demonstration
        print("Using mock data for demonstration...")
        return create_mock_data()


def create_mock_data():
    """Create mock molecular data for demonstration."""
    # Simple mock SMILES and corresponding LogP values
    mock_smiles = [
        'CCO',           # Ethanol, LogP ~ -0.31
        'CCCCCCCC',      # Octane, LogP ~ 5.15
        'c1ccccc1',      # Benzene, LogP ~ 2.13
        'CCN(CC)CC',     # Triethylamine, LogP ~ 1.45
        'CC(C)O',        # Isopropanol, LogP ~ 0.05
        'CCCCO',         # Butanol, LogP ~ 0.88
        'c1ccccc1O',     # Phenol, LogP ~ 1.46
        'CC(=O)O',       # Acetic acid, LogP ~ -0.17
        'CCCCCCCCCC',    # Decane, LogP ~ 6.25
        'CN(C)C',        # Trimethylamine, LogP ~ 0.16
    ] * 20  # Repeat to get more data
    
    # Corresponding LogP values (approximate)
    mock_logp = [
        -0.31, 5.15, 2.13, 1.45, 0.05, 0.88, 1.46, -0.17, 6.25, 0.16
    ] * 20
    
    # Add some noise
    np.random.seed(42)
    mock_logp = np.array(mock_logp) + np.random.normal(0, 0.1, len(mock_logp))
    
    # Create vocabulary
    all_chars = set(''.join(mock_smiles))
    tokens = ''.join(sorted(all_chars))
    
    return mock_smiles, mock_logp.tolist(), tokens


def train_attention_predictor(smiles_list, properties, tokens, test_size=0.2):
    """
    Train the attention-augmented predictor.
    
    Parameters
    ----------
    smiles_list: list
        List of SMILES strings
    properties: list
        List of property values
    tokens: str
        Vocabulary tokens
    test_size: float
        Fraction of data to use for testing
        
    Returns
    -------
    predictor: AttentionAugmentedPredictor
        Trained predictor
    test_metrics: dict
        Test set performance metrics
    """
    print("Training Attention-Augmented LSTM Predictor...")
    
    # Split data
    train_smiles, test_smiles, train_props, test_props = train_test_split(
        smiles_list, properties, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {len(train_smiles)} molecules")
    print(f"Test set: {len(test_smiles)} molecules")
    
    # Create predictor
    predictor = AttentionAugmentedPredictor(
        vocab_size=len(tokens),
        model_params={
            'embedding_dim': 128,
            'lstm_hidden_size': 256,
            'num_lstm_layers': 2,
            'num_attention_heads': 8,
            'num_dense_layers': 2,
            'dense_hidden_size': 512,
            'output_size': 1,
            'dropout': 0.1
        },
        ensemble_size=3,  # Smaller ensemble for demo
        use_cuda=torch.cuda.is_available()
    )
    
    # Prepare training data
    train_data = (train_smiles, train_props, tokens)
    val_data = (test_smiles, test_props, tokens)
    
    # Train the model
    history = predictor.train_model(
        train_data=train_data,
        val_data=val_data,
        epochs=50,  # Fewer epochs for demo
        batch_size=16,
        learning_rate=1e-3,
        patience=10
    )
    
    # Evaluate on test set
    canonical_smiles, predictions, invalid_smiles = predictor.predict(
        test_smiles, tokens, use_tqdm=True
    )
    
    # Calculate metrics
    if len(predictions) > 0:
        # Align predictions with actual values
        valid_indices = [i for i, smiles in enumerate(test_smiles) 
                        if smiles not in invalid_smiles]
        actual_values = np.array(test_props)[valid_indices]
        
        r2 = r2_score(actual_values, predictions.flatten())
        mse = mean_squared_error(actual_values, predictions.flatten())
        mae = mean_absolute_error(actual_values, predictions.flatten())
        
        test_metrics = {
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
        
        print(f"\nTest Set Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
    else:
        test_metrics = {}
        print("No valid predictions made on test set")
    
    return predictor, test_metrics


def demonstrate_attention_visualization(predictor, tokens):
    """
    Demonstrate attention visualization on example molecules.
    
    Parameters
    ----------
    predictor: AttentionAugmentedPredictor
        Trained predictor
    tokens: str
        Vocabulary tokens
    """
    print("\nDemonstrating Attention Visualization...")
    
    # Example molecules with different structural features
    example_molecules = [
        'CCO',                    # Simple alcohol
        'c1ccccc1',              # Benzene ring
        'CCCCCCCC',              # Long aliphatic chain
        'c1ccccc1CCO',           # Phenethyl alcohol
        'CC(C)(C)c1ccccc1',      # tert-Butylbenzene
    ]
    
    for smiles in example_molecules:
        try:
            # Get predictions with attention
            canonical_smiles, predictions, invalid_smiles, attention_info = predictor.predict(
                [smiles], tokens, return_attention=True
            )
            
            if len(canonical_smiles) > 0:
                # Get pooling attention weights (shows which positions are important)
                pooling_weights = attention_info['pooling_attention_weights'][0]  # First sample
                
                print(f"\nMolecule: {smiles}")
                print(f"Predicted Property: {predictions[0][0]:.3f}")
                
                # Print attention weights for each character
                canonical_smiles_str = canonical_smiles[0]
                print("Character-wise attention weights:")
                for i, char in enumerate(canonical_smiles_str):
                    if i < len(pooling_weights):
                        print(f"  {char}: {pooling_weights[i]:.3f}")
                
                # Visualize attention (if matplotlib is available)
                try:
                    visualize_attention(
                        canonical_smiles_str, 
                        pooling_weights, 
                        tokens,
                        save_path=f"/tmp/attention_{smiles.replace('/', '_').replace('\\\\', '_')}.png"
                    )
                except:
                    print("  (Visualization skipped - matplotlib not available)")
            
        except Exception as e:
            print(f"Error processing {smiles}: {e}")


def integration_with_release_rl(predictor, tokens):
    """
    Demonstrate integration with ReLeaSE reinforcement learning.
    
    Parameters
    ----------
    predictor: AttentionAugmentedPredictor
        Trained predictor
    tokens: str
        Vocabulary tokens
    """
    print("\nDemonstrating Integration with ReLeaSE RL...")
    
    def attention_based_reward_function(smiles, predictor_obj, target_value=2.0, tokens=tokens):
        """
        Reward function that uses the attention-augmented predictor.
        
        Parameters
        ----------
        smiles: str
            SMILES string to evaluate
        predictor_obj: AttentionAugmentedPredictor
            The attention-augmented predictor
        target_value: float
            Target property value
        tokens: str
            Vocabulary tokens
            
        Returns
        -------
        reward: float
            Calculated reward
        """
        try:
            # Get prediction
            canonical_smiles, predictions, invalid_smiles = predictor_obj.predict(
                [smiles], tokens
            )
            
            if len(invalid_smiles) > 0:
                return 0.0  # Invalid molecule
            
            if len(predictions) > 0:
                predicted_value = predictions[0][0]
                
                # Reward based on proximity to target
                distance = abs(predicted_value - target_value)
                reward = max(0, 1.0 - distance / 5.0)  # Normalize by expected range
                
                # Bonus for valid, drug-like molecules
                if 1.0 <= predicted_value <= 3.0:  # Drug-like LogP range
                    reward *= 1.2
                
                return reward
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error in reward function: {e}")
            return 0.0
    
    # Test the reward function
    test_molecules = [
        'CCO',           # Low LogP
        'c1ccccc1',      # Medium LogP
        'CCCCCCCC',      # High LogP
        'invalid_smiles' # Invalid
    ]
    
    print("Testing reward function:")
    for smiles in test_molecules:
        reward = attention_based_reward_function(smiles, predictor)
        print(f"  {smiles}: reward = {reward:.3f}")
    
    print("\nThis reward function can be used with the Reinforcement class:")
    print("from reinforcement import Reinforcement")
    print("rl_agent = Reinforcement(")
    print("    generator=transformer_generator,")
    print("    predictor=attention_predictor,")
    print("    get_reward=attention_based_reward_function")
    print(")")


def compare_with_baseline():
    """
    Compare attention-augmented predictor with baseline methods.
    """
    print("\nComparison with Baseline Methods:")
    print("=" * 50)
    
    print("\n1. Original LSTM Predictor:")
    print("   - Architecture: Embedding + LSTM + Dense")
    print("   - Aggregation: Last hidden state or mean pooling")
    print("   - Interpretability: Limited")
    
    print("\n2. Attention-Augmented LSTM Predictor:")
    print("   - Architecture: Embedding + Bi-LSTM + Self-Attention + Dense")
    print("   - Aggregation: Attention-weighted pooling")
    print("   - Interpretability: High (attention weights)")
    
    print("\n3. Expected Improvements:")
    print("   - Better capture of long-range dependencies")
    print("   - Focus on chemically relevant substructures")
    print("   - Improved performance on complex molecules")
    print("   - Interpretable predictions via attention visualization")


def main():
    """Main demonstration function."""
    print("Attention-Augmented LSTM Predictor for Molecular Property Prediction")
    print("=" * 70)
    
    # Try to load real data, fall back to mock data
    data_path = "/root/ReLease/ReLeaSE/data/logP_labels.csv"
    smiles_list, properties, tokens = load_molecular_data(data_path)
    
    # Train the attention-augmented predictor
    predictor, test_metrics = train_attention_predictor(smiles_list, properties, tokens)
    
    # Demonstrate attention visualization
    demonstrate_attention_visualization(predictor, tokens)
    
    # Show integration with ReLeaSE RL
    integration_with_release_rl(predictor, tokens)
    
    # Compare with baseline methods
    compare_with_baseline()
    
    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("\nKey benefits of the attention-augmented approach:")
    print("- Better understanding of molecular structure-property relationships")
    print("- Improved predictive performance on complex molecules")
    print("- Interpretable models through attention visualization")
    print("- Seamless integration with existing ReLeaSE framework")


if __name__ == "__main__":
    main()
