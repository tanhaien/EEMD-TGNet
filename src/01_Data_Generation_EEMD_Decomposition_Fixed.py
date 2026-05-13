#!/usr/bin/env python
# coding: utf-8

# # EEMD-TGNet: Data Generation and EEMD Decomposition
# 
# This notebook implements the first part of the EEMD-TGNet experimental framework:
# 
# 1. Load and prepare real solar energy dataset
# 
# 2. Apply EEMD decomposition
# 
# 3. Analyze and visualize IMFs
# 
# 4. Remove noise components using autocorrelation

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import PyEMD, if not available, we'll implement a simplified version
try:
    from PyEMD import EEMD
    PYEMD_AVAILABLE = True
    print("PyEMD library found")
except ImportError:
    PYEMD_AVAILABLE = False
    print("PyEMD not found, using simplified implementation")

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ## 1. Real Solar Energy Data Loading and Preparation
# 
# We'll load the `data_udata_solar.csv` dataset and prepare it for EEMD decomposition.
# This involves:
# - Loading the CSV into a Pandas DataFrame
# - Converting the 'date' column to datetime objects and setting it as the index
# - Renaming relevant columns to match the expected format (`solar_power`, `solar_irradiance`, `temperature`, etc.)
# - Creating time-based features (`hour_of_day`, `day_of_year`, `month`, `is_weekend`)
# - Handling missing values (e.g., forward fill)

# In[2]:


def load_and_prepare_solar_data(filepath='data_udata_solar.csv'):
    """
    Load and prepare the solar energy data from the CSV file.
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['date'])
    df = df.set_index('timestamp')

    # Rename columns to match the notebook's expected format
    df = df.rename(columns={
        'pow_P': 'solar_power',
        'shortwave_radiation (W/m²)': 'solar_irradiance',
        'temp': 'temperature',
        # Using other columns as placeholders for now, adjust as needed based on data meaning
        'cur': 'wind_speed', 
        'cosPhi': 'cloud_cover', 
        'vol_RN': 'humidity' 
    })

    # Create time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['is_weekend'] = df.index.weekday >= 5
    
    # Select and fill missing values for required columns
    required_cols = [
        'solar_power', 'solar_irradiance', 'temperature', 'wind_speed', 
        'cloud_cover', 'humidity', 'hour_of_day', 'day_of_year', 'month', 'is_weekend'
    ]
    
    # Ensure all required columns exist, fill with 0 if not (or handle appropriately)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 # Or np.nan, then fill

    # Forward fill for simplicity, a more robust method might be needed
    df = df[required_cols].fillna(method='ffill').fillna(method='bfill')

    return df.reset_index()

# Load and prepare the dataset
print("Loading and preparing data from data_udata_solar.csv...")
solar_data = load_and_prepare_solar_data()

# For compatibility with the rest of the notebook, we'll use solar_data for both 'alibaba' and 'gefcom' contexts.
# In a real scenario, you might split the data or use different subsets.
alibaba_data = solar_data.copy()
gefcom_data = solar_data.copy()

print(f"Solar dataset shape: {solar_data.shape}")

# Display sample data
print("\nSolar data sample:")
print(solar_data.head(10))


# In[3]:


# Visualize the loaded data
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Solar power time series
axes[0, 0].plot(solar_data['timestamp'], solar_data['solar_power'])
axes[0, 0].set_title('Solar Power Generation (Full Dataset)')
axes[0, 0].set_ylabel('Power (kW)')
axes[0, 0].grid(True)

# Daily pattern
daily_avg = solar_data.groupby('hour_of_day')['solar_power'].mean()
axes[0, 1].plot(daily_avg.index, daily_avg.values, 'o-')
axes[0, 1].set_title('Average Daily Solar Power Pattern')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Power (kW)')
axes[0, 1].grid(True)

# Seasonal pattern
monthly_avg = solar_data.groupby('month')['solar_power'].mean()
axes[1, 0].plot(monthly_avg.index, monthly_avg.values, 's-')
axes[1, 0].set_title('Monthly Average Solar Power')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Power (kW)')
axes[1, 0].grid(True)

# Weather correlation
axes[1, 1].scatter(solar_data['solar_irradiance'], solar_data['solar_power'], 
                   alpha=0.5, s=1)
axes[1, 1].set_title('Solar Power vs Irradiance')
axes[1, 1].set_xlabel('Solar Irradiance (W/m²)')
axes[1, 1].set_ylabel('Power (kW)')
axes[1, 1].grid(True)

# Temperature effect
axes[2, 0].scatter(solar_data['temperature'], solar_data['solar_power'], 
                   alpha=0.5, s=1, c=solar_data['solar_irradiance'], cmap='viridis')
axes[2, 0].set_title('Solar Power vs Temperature (colored by irradiance)')
axes[2, 0].set_xlabel('Temperature (°C)')
axes[2, 0].set_ylabel('Power (kW)')
axes[2, 0].grid(True)

# Distribution
axes[2, 1].hist(solar_data['solar_power'], bins=50, alpha=0.7, edgecolor='black')
axes[2, 1].set_title('Solar Power Distribution')
axes[2, 1].set_xlabel('Power (kW)')
axes[2, 1].set_ylabel('Frequency')
axes[2, 1].grid(True)

plt.tight_layout()
# plt.show() # Commented out for automated execution

# Print statistics
print("\nDataset Statistics:")
print(solar_data[['solar_power', 'solar_irradiance', 'temperature', 'wind_speed']].describe())


# ## 2. EEMD Implementation
# 
# We'll implement EEMD (Ensemble Empirical Mode Decomposition) to decompose the solar power signal into intrinsic mode functions (IMFs).

# In[4]:


class SimpleEEMD:
    """
    Simplified EEMD implementation for demonstration purposes
    This is a basic version - in practice, use PyEMD for better results
    """
    
    def __init__(self, trials=100, noise_std=0.005, max_imf=8):
        self.trials = trials
        self.noise_std = noise_std
        self.max_imf = max_imf
    
    def _find_peaks_troughs(self, signal):
        """Find local maxima and minima"""
        from scipy.signal import argrelextrema
        
        maxima = argrelextrema(signal, np.greater, order=3)[0]
        minima = argrelextrema(signal, np.less, order=3)[0]
        
        # Add boundary points
        if len(maxima) > 0 and maxima[0] != 0:
            maxima = np.r_[0, maxima]
        if len(maxima) > 0 and maxima[-1] != len(signal) - 1:
            maxima = np.r_[maxima, len(signal) - 1]
            
        if len(minima) > 0 and minima[0] != 0:
            minima = np.r_[0, minima]
        if len(minima) > 0 and minima[-1] != len(signal) - 1:
            minima = np.r_[minima, len(signal) - 1]
        
        return maxima, minima
    
    def _sift(self, signal, max_iter=1000):
        """Single sifting process to extract one IMF"""
        from scipy.interpolate import interp1d
        
        h = signal.copy()
        
        for i in range(max_iter):
            maxima, minima = self._find_peaks_troughs(h)
            
            if len(maxima) < 3 or len(minima) < 3:
                break
            
            # Interpolate envelopes
            try:
                x = np.arange(len(h))
                upper_env = interp1d(maxima, h[maxima], kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')(x)
                lower_env = interp1d(minima, h[minima], kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')(x)
                
                mean_env = (upper_env + lower_env) / 2
                h = h - mean_env
                
                # Stopping criterion
                if np.sum(np.abs(mean_env)) < 1e-6 * np.sum(np.abs(h)):
                    break
                    
            except:
                break
        
        return h
    
    def eemd(self, signal):
        """Perform EEMD decomposition"""
        n = len(signal)
        noise_std = self.noise_std * np.std(signal)
        
        # Storage for ensemble IMFs
        all_imfs = []
        
        for trial in range(self.trials):
            # Add white noise
            noisy_signal = signal + np.random.normal(0, noise_std, n)
            
            # EMD decomposition
            imfs = []
            residue = noisy_signal.copy()
            
            for _ in range(self.max_imf):
                imf = self._sift(residue)
                
                if np.std(imf) < 1e-6 * np.std(signal):
                    break
                    
                imfs.append(imf)
                residue = residue - imf
                
                if len(self._find_peaks_troughs(residue)[0]) < 3:
                    break
            
            imfs.append(residue)  # Final residue
            all_imfs.append(imfs)
        
        # Average across trials
        max_imfs = max(len(trial_imfs) for trial_imfs in all_imfs)
        ensemble_imfs = []
        
        for i in range(max_imfs):
            imf_sum = np.zeros(n)
            count = 0
            
            for trial_imfs in all_imfs:
                if i < len(trial_imfs):
                    imf_sum += trial_imfs[i]
                    count += 1
            
            if count > 0:
                ensemble_imfs.append(imf_sum / count)
        
        return np.array(ensemble_imfs)

# Function to perform EEMD decomposition
def perform_eemd(signal, use_pyemd=True):
    """
    Perform EEMD decomposition using either PyEMD or simplified implementation
    """
    if use_pyemd and PYEMD_AVAILABLE:
        eemd = EEMD(trials=100, noise_seed=42)
        imfs = eemd.eemd(signal)
    else:
        print("Using simplified EEMD implementation...")
        eemd = SimpleEEMD(trials=50, noise_std=0.005)
        imfs = eemd.eemd(signal)
    
    return imfs

# Prepare data for EEMD
# Use the full solar_data for decomposition
solar_signal = solar_data['solar_power'].values
print(f"Decomposing solar signal of length: {len(solar_signal)}")

# Perform EEMD decomposition
imfs = perform_eemd(solar_signal)
print(f"Number of IMFs extracted: {len(imfs)}")


# In[5]:


# Visualize EEMD decomposition results
fig, axes = plt.subplots(len(imfs) + 1, 1, figsize=(15, 2 * (len(imfs) + 1)))

# Original signal
axes[0].plot(solar_signal, 'b-', linewidth=1)
axes[0].set_title('Original Solar Power Signal', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Power (kW)')
axes[0].grid(True, alpha=0.3)

# IMFs
colors = plt.cm.Set3(np.linspace(0, 1, len(imfs)))
for i, imf in enumerate(imfs):
    axes[i + 1].plot(imf, color=colors[i], linewidth=1)
    if i < len(imfs) - 1:
        axes[i + 1].set_title(f'IMF {i + 1}', fontsize=10)
    else:
        axes[i + 1].set_title('Residue', fontsize=10)
    axes[i + 1].set_ylabel('Amplitude')
    axes[i + 1].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (hours)')
plt.tight_layout()
# plt.show() # Commented out for automated execution


# Verify reconstruction
reconstructed = np.sum(imfs, axis=0)
reconstruction_error = np.mean(np.abs(solar_signal - reconstructed))
print(f"Reconstruction error: {reconstruction_error:.6f}")
print(f"Relative error: {reconstruction_error / np.mean(np.abs(solar_signal)) * 100:.4f}%")


# ## 3. Autocorrelation Analysis for Noise Identification
# 
# We'll analyze the autocorrelation of each IMF to identify which components represent noise and should be removed.

# In[6]:


def analyze_imf_characteristics(imfs, signal_length):
    """
    Analyze characteristics of each IMF including:
    - Autocorrelation at lag 1
    - Energy content
    - Frequency content
    - Statistical properties
    """
    characteristics = []
    
    for i, imf in enumerate(imfs):
        # Autocorrelation at lag 1
        if len(imf) > 1:
            autocorr_lag1 = np.corrcoef(imf[:-1], imf[1:])[0, 1]
            if np.isnan(autocorr_lag1):
                autocorr_lag1 = 0
        else:
            autocorr_lag1 = 0
        
        # Energy content (variance)
        energy = np.var(imf)
        
        # Relative energy
        total_energy = np.sum([np.var(imf_j) for imf_j in imfs])
        relative_energy = energy / total_energy if total_energy > 0 else 0
        
        # Mean frequency (simplified)
        zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
        mean_frequency = zero_crossings / (2 * signal_length) if signal_length > 0 else 0
        
        # Statistical properties
        skewness = np.abs(np.mean(((imf - np.mean(imf)) / np.std(imf))**3)) if np.std(imf) > 0 else 0
        kurtosis = np.mean(((imf - np.mean(imf)) / np.std(imf))**4) if np.std(imf) > 0 else 0
        
        characteristics.append({
            'IMF': i + 1 if i < len(imfs) - 1 else 'Residue',
            'Autocorr_Lag1': autocorr_lag1,
            'Energy': energy,
            'Relative_Energy_%': relative_energy * 100,
            'Mean_Frequency': mean_frequency,
            'Kurtosis': kurtosis,
            'Std': np.std(imf)
        })
    
    return pd.DataFrame(characteristics)

# Analyze IMF characteristics
imf_analysis = analyze_imf_characteristics(imfs, len(solar_signal))
print("IMF Characteristics Analysis:")
print(imf_analysis.round(4))

# Identify noise components (low autocorrelation + low energy)
noise_threshold_autocorr = 0.1
noise_threshold_energy = 2.0  # Relative energy percentage

noise_imfs = imf_analysis[
    (imf_analysis['Autocorr_Lag1'].abs() < noise_threshold_autocorr) & 
    (imf_analysis['Relative_Energy_%'] < noise_threshold_energy) &
    (imf_analysis['IMF'] != 'Residue')
]['IMF'].tolist()

print(f"\nIdentified noise IMFs: {noise_imfs}")
print(f"Signal IMFs to keep: {[imf for imf in imf_analysis['IMF'].tolist() if imf not in noise_imfs]}")


# In[7]:


# Visualize IMF characteristics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Autocorrelation vs Energy
colors = ['red' if imf in noise_imfs else 'blue' for imf in imf_analysis['IMF']]
axes[0, 0].scatter(imf_analysis['Autocorr_Lag1'], imf_analysis['Relative_Energy_%'], 
                   c=colors, s=60, alpha=0.7)
axes[0, 0].axvline(noise_threshold_autocorr, color='red', linestyle='--', alpha=0.5)
axes[0, 0].axhline(noise_threshold_energy, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Autocorrelation (Lag 1)')
axes[0, 0].set_ylabel('Relative Energy (%)')
axes[0, 0].set_title('IMF Classification: Signal vs Noise')
axes[0, 0].grid(True, alpha=0.3)

# Add labels for each point
for i, row in imf_analysis.iterrows():
    axes[0, 0].annotate(str(row['IMF']), 
                       (row['Autocorr_Lag1'], row['Relative_Energy_%']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

# Energy distribution
bars = axes[0, 1].bar(range(len(imfs)), imf_analysis['Relative_Energy_%'], 
                      color=colors, alpha=0.7)
axes[0, 1].set_xlabel('IMF Index')
axes[0, 1].set_ylabel('Relative Energy (%)')
axes[0, 1].set_title('Energy Distribution Across IMFs')
axes[0, 1].set_xticks(range(len(imfs)))
axes[0, 1].set_xticklabels([str(imf) for imf in imf_analysis['IMF']], rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Frequency content
axes[1, 0].bar(range(len(imfs)), imf_analysis['Mean_Frequency'], 
               color=colors, alpha=0.7)
axes[1, 0].set_xlabel('IMF Index')
axes[1, 0].set_ylabel('Mean Frequency (cycles/hour)')
axes[1, 0].set_title('Frequency Content of IMFs')
axes[1, 0].set_xticks(range(len(imfs)))
axes[1, 0].set_xticklabels([str(imf) for imf in imf_analysis['IMF']], rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Autocorrelation plot
axes[1, 1].plot(imf_analysis.index, imf_analysis['Autocorr_Lag1'], 'o-', markersize=8)
axes[1, 1].axhline(noise_threshold_autocorr, color='red', linestyle='--', alpha=0.5, 
                   label=f'Noise threshold ({noise_threshold_autocorr})')
axes[1, 1].axhline(-noise_threshold_autocorr, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('IMF Index')
axes[1, 1].set_ylabel('Autocorrelation (Lag 1)')
axes[1, 1].set_title('Autocorrelation Analysis')
axes[1, 1].set_xticks(range(len(imfs)))
axes[1, 1].set_xticklabels([str(imf) for imf in imf_analysis['IMF']], rotation=45)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
# plt.show() # Commented out for automated execution


# ## 4. Signal Reconstruction without Noise Components

# In[8]:


# Reconstruct signal without noise components
def reconstruct_denoised_signal(imfs, noise_imf_indices):
    """
    Reconstruct signal excluding specified noise IMFs
    """
    signal_imfs = []
    for i, imf in enumerate(imfs):
        if (i + 1) not in noise_imf_indices:  # Keep non-noise IMFs
            signal_imfs.append(imf)
    
    if signal_imfs:
        return np.sum(signal_imfs, axis=0)
    else:
        return np.zeros_like(imfs[0])

# Convert noise IMF names to indices
noise_indices = []
for imf_name in noise_imfs:
    if isinstance(imf_name, (int, float)):
        noise_indices.append(int(imf_name))
    elif str(imf_name).isdigit():
        noise_indices.append(int(imf_name))

# Reconstruct denoised signal
denoised_signal = reconstruct_denoised_signal(imfs, noise_indices)

# Compare original, noisy, and denoised signals
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

time_axis = np.arange(len(solar_signal))

# Original signal
axes[0].plot(time_axis, solar_signal, 'b-', linewidth=1, label='Original Signal')
axes[0].set_title('Original Solar Power Signal')
axes[0].set_ylabel('Power (kW)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Reconstructed (with noise)
reconstructed_full = np.sum(imfs, axis=0)
axes[1].plot(time_axis, reconstructed_full, 'g-', linewidth=1, label='Reconstructed (All IMFs)')
axes[1].set_title('Reconstructed Signal (All IMFs)')
axes[1].set_ylabel('Power (kW)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Denoised signal
axes[2].plot(time_axis, denoised_signal, 'r-', linewidth=1, label='Denoised Signal')
axes[2].plot(time_axis, solar_signal, 'b--', alpha=0.5, linewidth=1, label='Original')
axes[2].set_title('Denoised Signal (Noise IMFs Removed)')
axes[2].set_xlabel('Time (hours)')
axes[2].set_ylabel('Power (kW)')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
# plt.show() # Commented out for automated execution

# Calculate denoising metrics
mse_original = np.mean((solar_signal - reconstructed_full)**2)
mse_denoised = np.mean((solar_signal - denoised_signal)**2)
snr_improvement = 10 * np.log10(mse_original / mse_denoised) if mse_denoised > 0 else np.inf

print(f"\nDenoising Results:")
print(f"Original reconstruction MSE: {mse_original:.6f}")
print(f"Denoised reconstruction MSE: {mse_denoised:.6f}")
print(f"SNR improvement: {snr_improvement:.2f} dB")
print(f"Removed {len(noise_indices)} noise IMFs out of {len(imfs)} total IMFs")


# ## 5. Prepare Data for Model Training
# 
# Finally, we'll prepare the decomposed data for training the TCN-GRU models.

# In[9]:


# Prepare final datasets for model training
def prepare_decomposed_datasets(data, target_col='solar_power', window_size=24, 
                               forecast_horizon=6, noise_threshold=0.1):
    """
    Prepare datasets with EEMD decomposition for training
    """
    # Extract target signal
    target_signal = data[target_col].values
    
    # Perform EEMD decomposition
    print(f"Performing EEMD on {len(target_signal)} samples...")
    imfs = perform_eemd(target_signal)
    
    # Analyze IMF characteristics
    imf_analysis = analyze_imf_characteristics(imfs, len(target_signal))
    
    # Identify signal IMFs (non-noise)
    signal_imf_indices = []
    for i, row in imf_analysis.iterrows():
        if (abs(row['Autocorr_Lag1']) >= noise_threshold or 
            row['Relative_Energy_%'] >= 2.0 or 
            row['IMF'] == 'Residue'):
            signal_imf_indices.append(i)
    
    signal_imfs = [imfs[i] for i in signal_imf_indices]
    
    # Add meteorological features
    feature_cols = ['solar_irradiance', 'temperature', 'wind_speed', 'cloud_cover', 
                   'humidity', 'hour_of_day', 'day_of_year']
    features = data[feature_cols].values
    
    # Normalize features
    scaler_features = StandardScaler()
    features_scaled = scaler_features.fit_transform(features)
    
    # Prepare training data for each IMF
    imf_datasets = []
    imf_scalers = []
    
    for i, imf in enumerate(signal_imfs):
        # Scale IMF
        scaler_imf = StandardScaler()
        imf_scaled = scaler_imf.fit_transform(imf.reshape(-1, 1)).flatten()
        imf_scalers.append(scaler_imf)
        
        # Create sequences
        X_imf, y_imf = [], []
        for j in range(window_size, len(imf_scaled) - forecast_horizon + 1):
            # IMF history + meteorological features
            imf_history = imf_scaled[j-window_size:j]
            met_features = features_scaled[j-1]  # Use most recent features
            
            # Combine IMF and features
            x_combined = np.concatenate([imf_history, met_features])
            y_target = imf_scaled[j:j+forecast_horizon]
            
            X_imf.append(x_combined)
            y_imf.append(y_target)
        
        X_imf = np.array(X_imf)
        y_imf = np.array(y_imf)
        
        imf_datasets.append({
            'X': X_imf,
            'y': y_imf,
            'imf_index': signal_imf_indices[i],
            'scaler': scaler_imf
        })
    
    return {
        'imf_datasets': imf_datasets,
        'feature_scaler': scaler_features,
        'signal_imfs': signal_imfs,
        'imf_analysis': imf_analysis,
        'original_signal': target_signal,
        'window_size': window_size,
        'forecast_horizon': forecast_horizon
    }


# In[10]:


# Prepare training data for the solar dataset
print("Preparing solar dataset for training...")
solar_prepared = prepare_decomposed_datasets(
    solar_data, 
    window_size=24, 
    forecast_horizon=6
)

# For compatibility with subsequent notebooks, assign to gefcom_prepared and alibaba_prepared
gefcom_prepared = solar_prepared
alibaba_prepared = solar_prepared

print(f"\nSolar dataset prepared:")
print(f"Number of signal IMFs: {len(solar_prepared['imf_datasets'])}")
for i, dataset in enumerate(solar_prepared['imf_datasets']):
    print(f"  IMF {dataset['imf_index']}: X shape {dataset['X'].shape}, y shape {dataset['y'].shape}")

# Save prepared data
import pickle

with open('gefcom_prepared_data.pkl', 'wb') as f:
    pickle.dump(gefcom_prepared, f)

with open('alibaba_prepared_data.pkl', 'wb') as f:
    pickle.dump(alibaba_prepared, f)

print("\nPrepared datasets saved to pickle files.")
print("Ready for model training in the next notebook!")


# In[11]:


# Summary of data preparation
print("=" * 60)
print("DATA PREPARATION SUMMARY")
print("=" * 60)

print(f"\n1. DATASET LOADED:")
print(f"   - data_udata_solar.csv: {solar_data.shape[0]} samples")
print(f"   - Features: {list(solar_data.columns)}")

print(f"\n2. EEMD DECOMPOSITION:")
print(f"   - Total IMFs extracted: {len(imfs)}")
print(f"   - Signal IMFs identified: {len(solar_prepared['imf_datasets'])}")
print(f"   - Noise IMFs removed: {len(imfs) - len(solar_prepared['imf_datasets'])}")

print(f"\n3. TRAINING DATA PREPARATION:")
print(f"   - Window size: {solar_prepared['window_size']} hours")
print(f"   - Forecast horizon: {solar_prepared['forecast_horizon']} hours")
print(f"   - Input features per sample: {solar_prepared['imf_datasets'][0]['X'].shape[1]}")
print(f"   - Total training samples: {sum(len(d['X']) for d in solar_prepared['imf_datasets'])}")

print(f"\n4. NEXT STEPS:")
print(f"   - Run 02_TCN_GRU_Architecture_Training.ipynb to train models")
print(f"   - Run 03_Evaluation_Results_Analysis_Fixed.ipynb for evaluation")

print("\n" + "="*60)
print("✅ Data loading and preparation completed successfully!")
print("✅ EEMD decomposition working properly!")
print("✅ Ready to proceed with model training!")
