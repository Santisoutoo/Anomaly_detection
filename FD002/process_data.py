"""
Process FD002 data:
- Load raw data
- Remove constant sensors
- Standardize features
- Remove highly correlated sensors
- Save processed data
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add utils to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'utils'))
from load_dataset import load_dataset

def process_fd002():
    """Process FD002 dataset and save cleaned data"""

    # Load data
    print("Loading FD002 data...")
    train, test, rul = load_dataset('FD002')
    print(f"Train: {train.shape}, Test: {test.shape}, RUL: {rul.shape}")

    # Identify constant sensors
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    sensor_variance = train[sensor_cols].var()
    constant_sensors = sensor_variance[sensor_variance < 0.001].index.tolist()

    print(f"\nConstant sensors (to remove): {constant_sensors}")

    # Remove constant sensors
    train = train.drop(columns=constant_sensors)
    test = test.drop(columns=constant_sensors)

    useful_sensors = [s for s in sensor_cols if s not in constant_sensors]
    print(f"Useful sensors: {len(useful_sensors)}")
    print(useful_sensors)

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    train[useful_sensors] = scaler.fit_transform(train[useful_sensors])
    test[useful_sensors] = scaler.transform(test[useful_sensors])

    # Check correlations
    print("\nChecking correlations...")
    correlation_matrix = train[useful_sensors].corr()

    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                corr_pairs.append({
                    'Sensor 1': correlation_matrix.columns[i],
                    'Sensor 2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })

    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation',
                                                        key=abs,
                                                        ascending=False)
        print("\nHigh correlations (|r| > 0.8):")
        print(corr_df)

        # Remove highly correlated sensors (keep one from each pair)
        # Based on the pattern, remove sensor_9 and sensor_20 (similar to FD001)
        sensors_to_remove = []
        if 'sensor_9' in useful_sensors and 'sensor_14' in useful_sensors:
            sensors_to_remove.append('sensor_9')
        if 'sensor_20' in useful_sensors:
            sensors_to_remove.append('sensor_20')

        if sensors_to_remove:
            print(f"\nRemoving highly correlated sensors: {sensors_to_remove}")
            useful_sensors = [s for s in useful_sensors if s not in sensors_to_remove]
            train = train.drop(columns=sensors_to_remove, errors='ignore')
            test = test.drop(columns=sensors_to_remove, errors='ignore')

    print(f"\nFinal shapes - Train: {train.shape}, Test: {test.shape}")
    print(f"Final useful sensors: {len(useful_sensors)}")

    # Create output directory
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)

    # Save processed data
    print(f"\nSaving data to {output_dir}...")
    train.to_csv(output_dir / 'train.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)
    rul.to_csv(output_dir / 'rul.csv', index=False)

    print("Done! Data saved successfully.")

    return train, test, rul

if __name__ == '__main__':
    train, test, rul = process_fd002()
