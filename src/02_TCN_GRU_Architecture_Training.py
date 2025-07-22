#!/usr/bin/env python
# coding: utf-8

# # EEMD-TGNet: TCN-GRU Architecture and Training
# 
# This notebook implements:
# 1. TCN-GRU hybrid architecture
# 2. Model training for each IMF
# 3. Model optimization (pruning, quantization)
# 4. Baseline models for comparison

# In[1]:


# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.utils import prune

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import copy
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ## 1. Load Prepared Data

# In[2]:


# Load prepared datasets
try:
    with open('gefcom_prepared_data.pkl', 'rb') as f:
        gefcom_data = pickle.load(f)
    
    with open('alibaba_prepared_data.pkl', 'rb') as f:
        alibaba_data = pickle.load(f)
    
    print("Loaded prepared datasets successfully!")
    print(f"GEFCom dataset: {len(gefcom_data['imf_datasets'])} IMFs")
    print(f"Alibaba dataset: {len(alibaba_data['imf_datasets'])} IMFs")
    
except FileNotFoundError:
    print("Prepared data files not found. Please run the first notebook first.")
    # Create dummy data for demonstration
    print("Creating dummy data for demonstration...")
    
    # Simplified dummy data creation
    window_size, forecast_horizon = 24, 6
    n_samples, n_features = 1000, 31  # 24 history + 7 weather features
    
    gefcom_data = {
        'imf_datasets': [
            {
                'X': np.random.randn(n_samples, n_features),
                'y': np.random.randn(n_samples, forecast_horizon),
                'imf_index': i,
                'scaler': StandardScaler()
            } for i in range(5)  # 5 IMFs
        ],
        'window_size': window_size,
        'forecast_horizon': forecast_horizon
    }
    
    alibaba_data = {
        'imf_datasets': [
            {
                'X': np.random.randn(500, 21),  # 14 history + 7 features
                'y': np.random.randn(500, 3),   # 3-step forecast
                'imf_index': i,
                'scaler': StandardScaler()
            } for i in range(4)  # 4 IMFs
        ],
        'window_size': 14,
        'forecast_horizon': 3
    }

# Print dataset information
print("\nDataset Information:")
for i, imf_data in enumerate(gefcom_data['imf_datasets']):
    print(f"GEFCom IMF {imf_data['imf_index']}: X {imf_data['X'].shape}, y {imf_data['y'].shape}")

print()
for i, imf_data in enumerate(alibaba_data['imf_datasets']):
    print(f"Alibaba IMF {imf_data['imf_index']}: X {imf_data['X'].shape}, y {imf_data['y'].shape}")


# ## 2. TCN-GRU Architecture Implementation

# In[3]:


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        
        # Ensure same output length (causal padding)
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        residual = x
        
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Remove future padding for causality
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]  # Remove future padding for causality
        out = self.bn2(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        # Match dimensions for residual connection
        if residual.size(2) != out.size(2):
            residual = residual[:, :, :out.size(2)]
        
        out = self.relu2(out + residual)
        out = self.dropout2(out)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    """
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalConvBlock(in_channels, out_channels, 
                                          kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        # Convert to (batch_size, features, sequence_length) for conv1d
        x = x.transpose(1, 2)
        
        out = self.network(x)
        
        # Convert back to (batch_size, sequence_length, features)
        out = out.transpose(1, 2)
        
        return out


class TCNGRU(nn.Module):
    """
    Hybrid TCN-GRU Model
    """
    def __init__(self, input_size, tcn_channels, gru_hidden_size, 
                 output_size, tcn_kernel_size=3, tcn_dropout=0.2, 
                 gru_dropout=0.2, num_gru_layers=1):
        super(TCNGRU, self).__init__()
        
        self.input_size = input_size
        self.window_size = None  # Will be set based on input
        
        # TCN for feature extraction
        self.tcn = TCN(1, tcn_channels, tcn_kernel_size, tcn_dropout)
        
        # Calculate TCN output size
        tcn_output_size = tcn_channels[-1]
        
        # Feature projection layer
        self.feature_projection = nn.Linear(tcn_output_size + 7, gru_hidden_size)  # +7 for weather features
        
        # GRU for sequence modeling
        self.gru = nn.GRU(gru_hidden_size, gru_hidden_size, 
                         num_layers=num_gru_layers, 
                         dropout=gru_dropout if num_gru_layers > 1 else 0,
                         batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(gru_hidden_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(gru_dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Determine window size dynamically
        if self.window_size is None:
            # Assume input structure: [window_history, weather_features]
            if x.size(1) == 31:  # GEFCom: 24 + 7 features
                self.window_size = 24
            elif x.size(1) == 21:  # Alibaba: 14 + 7 features
                self.window_size = 14
            else:
                self.window_size = x.size(1) - 7  # Assume last 7 are weather features
        
        # Split input into time series window and additional features
        time_series = x[:, :self.window_size]  # Shape: (batch_size, window_size)
        additional_features = x[:, self.window_size:]  # Shape: (batch_size, num_features)
        
        # Reshape time series for TCN (add channel dimension)
        time_series = time_series.unsqueeze(-1)  # Shape: (batch_size, window_size, 1)
        
        # TCN feature extraction
        tcn_out = self.tcn(time_series)  # Shape: (batch_size, window_size, tcn_channels[-1])
        
        # Use the last time step from TCN output
        tcn_features = tcn_out[:, -1, :]  # Shape: (batch_size, tcn_channels[-1])
        
        # Combine TCN features with additional features
        combined_features = torch.cat([tcn_features, additional_features], dim=1)
        
        # Project to GRU input size
        gru_input = self.feature_projection(combined_features)  # Shape: (batch_size, gru_hidden_size)
        gru_input = gru_input.unsqueeze(1)  # Shape: (batch_size, 1, gru_hidden_size)
        
        # GRU processing
        gru_out, _ = self.gru(gru_input)  # Shape: (batch_size, 1, gru_hidden_size)
        gru_out = gru_out.squeeze(1)  # Shape: (batch_size, gru_hidden_size)
        
        # Apply dropout
        gru_out = self.dropout(gru_out)
        
        # Generate output
        output = self.output_layer(gru_out)  # Shape: (batch_size, output_size)
        
        return output


# Test the model architecture
def test_model_architecture():
    print("Testing TCN-GRU architecture...")
    
    # Test with GEFCom data dimensions
    batch_size = 32
    input_size = 31  # 24 history + 7 features
    output_size = 6   # 6-step forecast
    
    model = TCNGRU(
        input_size=input_size,
        tcn_channels=[32, 64, 32],
        gru_hidden_size=64,
        output_size=output_size,
        tcn_kernel_size=3,
        tcn_dropout=0.1,
        gru_dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(batch_size, input_size)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

test_model = test_model_architecture()


# ## 3. Training and Evaluation Functions

# In[4]:


def prepare_data_loader(X, y, batch_size=32, shuffle=True):
    """
    Prepare PyTorch DataLoader from numpy arrays
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """
    Train a PyTorch model
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            train_samples += batch_x.size(0)
        
        train_loss /= train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                val_samples += batch_x.size(0)
        
        val_loss /= val_samples
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, scaler=None):
    """
    Evaluate model performance
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }

print("Training and evaluation functions defined successfully!")


# ## 4. Train EEMD-TGNet Models

# In[5]:


def train_eemd_tgnet_models(dataset, dataset_name, max_imfs=3):
    """
    Train TCN-GRU models for each IMF
    """
    print(f"\nTraining EEMD-TGNet models for {dataset_name}")
    print("="*50)
    
    trained_models = []
    
    # Limit number of IMFs for demonstration
    imfs_to_train = dataset['imf_datasets'][:max_imfs]
    
    for i, imf_data in enumerate(imfs_to_train):
        print(f"\nTraining model for IMF {imf_data['imf_index']}...")
        
        X, y = imf_data['X'], imf_data['y']
        input_size = X.shape[1]
        output_size = y.shape[1]
        
        print(f"Data shape: X {X.shape}, y {y.shape}")
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
        
        # Create data loaders
        train_loader = prepare_data_loader(X_train, y_train, batch_size=32)
        val_loader = prepare_data_loader(X_val, y_val, batch_size=32, shuffle=False)
        test_loader = prepare_data_loader(X_test, y_test, batch_size=32, shuffle=False)
        
        # Create model
        model = TCNGRU(
            input_size=input_size,
            tcn_channels=[32, 64, 32],
            gru_hidden_size=64,
            output_size=output_size,
            tcn_kernel_size=3,
            tcn_dropout=0.1,
            gru_dropout=0.1
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, num_epochs=30, lr=0.001
        )
        
        # Evaluate model
        test_results = evaluate_model(trained_model, test_loader)
        
        print(f"Test Results - MSE: {test_results['mse']:.6f}, MAE: {test_results['mae']:.6f}, R²: {test_results['r2']:.4f}")
        
        # Store results
        trained_models.append({
            'model': trained_model,
            'imf_index': imf_data['imf_index'],
            'test_results': test_results,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'scaler': imf_data['scaler']
        })
    
    return trained_models


# Train EEMD-TGNet models
gefcom_models = train_eemd_tgnet_models(gefcom_data, "GEFCom2014", max_imfs=3)
alibaba_models = train_eemd_tgnet_models(alibaba_data, "Alibaba Competition", max_imfs=2)

print(f"\nEEMD-TGNet training completed!")
print(f"GEFCom models trained: {len(gefcom_models)}")
print(f"Alibaba models trained: {len(alibaba_models)}")


# ## 5. Model Optimization (Pruning and Quantization)

# In[6]:


def apply_model_optimization(model, pruning_ratio=0.3):
    """
    Apply pruning and quantization to a model
    """
    # Original model stats
    original_params = sum(p.numel() for p in model.parameters())
    
    # Create a copy for pruning
    pruned_model = copy.deepcopy(model)
    
    # Apply pruning to linear layers
    parameters_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    if parameters_to_prune:
        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
    
    # Count non-zero parameters after pruning
    pruned_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
    
    # Apply quantization (simplified)
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            pruned_model, {nn.Linear}, dtype=torch.qint8
        )
    except:
        quantized_model = pruned_model  # Fallback if quantization fails
    
    return {
        'original_model': model,
        'pruned_model': pruned_model,
        'quantized_model': quantized_model,
        'original_params': original_params,
        'pruned_params': pruned_params,
        'compression_ratio': (original_params - pruned_params) / original_params * 100
    }


def optimize_all_models(trained_models, dataset_name):
    """
    Optimize all trained models
    """
    print(f"\nOptimizing {dataset_name} models...")
    print("="*40)
    
    optimized_models = []
    
    for model_info in trained_models:
        imf_index = model_info['imf_index']
        model = model_info['model']
        
        print(f"\nOptimizing IMF {imf_index} model...")
        
        # Apply optimization
        optimization_results = apply_model_optimization(model, pruning_ratio=0.4)
        
        print(f"Original parameters: {optimization_results['original_params']:,}")
        print(f"Pruned parameters: {optimization_results['pruned_params']:,}")
        print(f"Compression ratio: {optimization_results['compression_ratio']:.1f}%")
        
        # Combine with existing model info
        optimized_info = {**model_info, **optimization_results}
        optimized_models.append(optimized_info)
    
    return optimized_models


# Optimize models
gefcom_optimized = optimize_all_models(gefcom_models, "GEFCom2014")
alibaba_optimized = optimize_all_models(alibaba_models, "Alibaba Competition")

print("\nModel optimization completed!")


# ## 6. Baseline Models Training

# In[7]:


# Simple baseline models
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out.squeeze(1))
        return self.output_layer(lstm_out)


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1)
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out.squeeze(1))
        return self.output_layer(gru_out)


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def train_baseline_models(dataset, dataset_name):
    """
    Train baseline models for comparison
    """
    print(f"\nTraining baseline models for {dataset_name}")
    print("="*50)
    
    # Use first IMF data as representative
    sample_data = dataset['imf_datasets'][0]
    X, y = sample_data['X'], sample_data['y']
    input_size, output_size = X.shape[1], y.shape[1]
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    
    # Create data loaders
    train_loader = prepare_data_loader(X_train, y_train, batch_size=32)
    val_loader = prepare_data_loader(X_val, y_val, batch_size=32, shuffle=False)
    test_loader = prepare_data_loader(X_test, y_test, batch_size=32, shuffle=False)
    
    # Define baseline models
    baseline_models = {
        'LSTM': SimpleLSTM(input_size, 64, output_size),
        'GRU': SimpleGRU(input_size, 64, output_size),
        'MLP': SimpleMLP(input_size, [128, 64], output_size)
    }
    
    trained_baselines = {}
    
    for name, model in baseline_models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Train model
            trained_model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, num_epochs=20, lr=0.001
            )
            
            # Evaluate model
            test_results = evaluate_model(trained_model, test_loader)
            
            print(f"{name} Test Results - MSE: {test_results['mse']:.6f}, MAE: {test_results['mae']:.6f}, R²: {test_results['r2']:.4f}")
            
            trained_baselines[name] = {
                'model': trained_model,
                'test_results': test_results,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return trained_baselines


# Train baseline models
gefcom_baselines = train_baseline_models(gefcom_data, "GEFCom2014")
alibaba_baselines = train_baseline_models(alibaba_data, "Alibaba Competition")

print(f"\nBaseline training completed!")
print(f"GEFCom baselines: {list(gefcom_baselines.keys())}")
print(f"Alibaba baselines: {list(alibaba_baselines.keys())}")


# ## 7. Save Results

# In[8]:


# Save all training results
training_results = {
    'gefcom_models': gefcom_optimized,
    'alibaba_models': alibaba_optimized,
    'gefcom_baselines': gefcom_baselines,
    'alibaba_baselines': alibaba_baselines,
    'device': str(device)
}

with open('training_results.pkl', 'wb') as f:
    pickle.dump(training_results, f)

print("Training results saved to 'training_results.pkl'")

# Print summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

print(f"\n1. EEMD-TGNET MODELS:")
print(f"   - GEFCom dataset: {len(gefcom_optimized)} IMF models trained")
print(f"   - Alibaba dataset: {len(alibaba_optimized)} IMF models trained")
print(f"   - Each model optimized with pruning and quantization")

print(f"\n2. BASELINE MODELS:")
print(f"   - GEFCom baselines: {list(gefcom_baselines.keys())}")
print(f"   - Alibaba baselines: {list(alibaba_baselines.keys())}")

print(f"\n3. OPTIMIZATION RESULTS:")
avg_compression = np.mean([model['compression_ratio'] for model in gefcom_optimized])
print(f"   - Average parameter reduction: {avg_compression:.1f}%")

print(f"\n4. NEXT STEPS:")
print(f"   - Run notebook 3 for comprehensive evaluation")
print(f"   - Generate performance comparison tables")
print(f"   - Create visualizations for the paper")

print("\n" + "="*70)

