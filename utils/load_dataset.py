from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'

# Column name
index_names = ['unit_id', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names


def load_dataset(dataset_name, data_dir=None):
    """Load a dataset give its name"""
    
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / f'train_{dataset_name}.txt'
    test_path = data_dir / f'test_{dataset_name}.txt'
    rul_path = data_dir / f'RUL_{dataset_name}.txt'

    train_df = pd.read_csv(
        train_path,
        sep=r'\s+',
        header=None,
        names=col_names,
        usecols=range(26)
    )
    test_df = pd.read_csv(
        test_path,
        sep=r'\s+',
        header=None,
        names=col_names,
        usecols=range(26)
    )
    rul_df = pd.read_csv(
        rul_path,
        sep=r'\s+',
        header=None,
        names=['RUL']
    )

    # Calculate RUL
    train_df['RUL'] = train_df.groupby('unit_id')['time_cycles'].transform('max') - train_df['time_cycles']

    return train_df, test_df, rul_df

