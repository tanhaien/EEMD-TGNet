import numpy as np
import pandas as pd
import importlib.util
import sys
import pytest

# Dynamically import the module
spec = importlib.util.spec_from_file_location("data_gen", "src/01_Data_Generation_EEMD_Decomposition_Fixed.py")
data_gen = importlib.util.module_from_spec(spec)
sys.modules["data_gen"] = data_gen
spec.loader.exec_module(data_gen)

def test_analyze_imf_characteristics_keys_and_values():
    # Setup mock IMFs and signal length
    np.random.seed(42)
    # create three mock IMFs, each of length 10
    imf1 = np.random.normal(0, 1, 10)
    imf2 = np.random.normal(0, 0.5, 10)
    imf3 = np.random.normal(0, 0.1, 10) # Residu

    imfs = [imf1, imf2, imf3]
    signal_length = 10

    # Run the function
    df = data_gen.analyze_imf_characteristics(imfs, signal_length)

    # Verify return type
    assert isinstance(df, pd.DataFrame), "Result must be a pandas DataFrame"

    # Verify expected columns (keys)
    expected_columns = [
        'IMF', 'Autocorr_Lag1', 'Energy', 'Relative_Energy_%',
        'Mean_Frequency', 'Kurtosis', 'Std'
    ]
    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' is missing from the output DataFrame"

    # Verify the number of rows matches the number of IMFs
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    # Verify exact computed characteristics for the first IMF
    # Energy is variance
    expected_energy_1 = np.var(imf1)
    assert np.isclose(df.loc[0, 'Energy'], expected_energy_1), "Energy (variance) incorrectly computed"

    # Std is standard deviation
    expected_std_1 = np.std(imf1)
    assert np.isclose(df.loc[0, 'Std'], expected_std_1), "Standard deviation incorrectly computed"

    # Verify relative energy totals 100%
    assert np.isclose(df['Relative_Energy_%'].sum(), 100.0), "Relative energies should sum to 100%"

    # Verify the last row is named 'Residue' if multiple IMFs
    assert df.loc[len(df)-1, 'IMF'] == 'Residue', "The last IMF should be labelled 'Residue'"

def test_analyze_imf_characteristics_edge_cases():
    # Empty IMF list
    imfs = []
    signal_length = 10
    df = data_gen.analyze_imf_characteristics(imfs, signal_length)
    assert len(df) == 0, "Expected empty DataFrame for empty input"

    # Single IMF
    imf1 = np.array([1, 2, 3, 4])
    imfs = [imf1]
    df = data_gen.analyze_imf_characteristics(imfs, signal_length=4)
    assert len(df) == 1
    assert df.loc[0, 'IMF'] == 'Residue', "A single IMF should be labelled 'Residue'"

    # Very short IMFs
    imfs = [np.array([1.0]), np.array([2.0])]
    df = data_gen.analyze_imf_characteristics(imfs, signal_length=1)
    # Energy should be 0 since variance of length 1 is 0
    assert df.loc[0, 'Energy'] == 0.0
    # Autocorr should be 0
    assert df.loc[0, 'Autocorr_Lag1'] == 0.0