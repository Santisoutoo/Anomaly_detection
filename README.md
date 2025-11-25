# Anomaly Detection - NASA CMAPSS Dataset

Anomaly detection project for turbofan engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. This project implements and compares multiple anomaly detection techniques: classical methods, unsupervised learning, and deep learning.

## Dataset Description

The C-MAPSS dataset contains turbofan engine degradation simulation data with multivariate time series from sensors. It includes 4 sub-datasets (FD001-FD004) with different operating conditions and failure modes.

## Project Structure

```
Anomaly_detection/
├── data/                         # Raw CMAPSS dataset
│   ├── train_FD001.txt           # Training data
│   ├── test_FD001.txt            # Test data
│   ├── RUL_FD001.txt             # Remaining Useful Life (ground truth)
│   └── ...                       # FD002, FD003, FD004
│
├── utils/                         # Utilities and helper functions
│   ├── load_dataset.py           # Data loading and preprocessing
│   ├── metrics.py                # Evaluation metrics
│   └── plots.py                  # Visualization functions
│
├── FD001/                         # FD001 experiments
│   ├── 01_exploration.ipynb      # Exploratory Data Analysis
│   ├── data/                     # Processed data
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── rul.csv
│   │
│   ├── clasic_methods/           # Classical statistical methods
│   │   ├── z-score.ipynb         # Z-score based detection
│   │   ├── PCA.ipynb             # Principal Component Analysis
│   │   └── outputs/              # Results and plots
│   │
│   ├── unsupervised_learning/    # Unsupervised learning methods
│   │   ├── isolation_forest.ipynb
│   │   ├── One_Class_SVM.ipynb
│   │   └── outputs/
│   │
│   └── deep_learning/            # Deep Learning approaches
│       ├── Autoencoder.ipynb     # Basic Autoencoder
│       ├── LSTM_autoencoder.ipynb # LSTM Autoencoder
│       ├── TCN-VAE.ipynb         # Temporal Convolutional Network + VAE
│       └── outputs/
│
├── FD002/                         # Same structure for FD002
├── FD003/                         # Same structure for FD003
├── FD004/                         # Same structure for FD004
│
├── data_extraction.py             # Script to download dataset
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Installation

### Option 1: Using venv

```bash
# Clone the repository
git clone <repository-url>
cd Anomaly_detection

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Using uv

[uv](https://github.com/astral-sh/uv) is an ultra-fast Python package manager written in Rust.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd Anomaly_detection

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies with uv (much faster)
uv pip install -r requirements.txt
```

## Download the Dataset

The project uses the **CMAPSS Jet Engine Simulated Data** dataset from Kaggle. There are **two ways** to download it:

### Method 1: Automatic Script

```bash
# Make sure your virtual environment is activated
python data_extraction.py
```

This script will:
- Automatically create the `data/` directory if it doesn't exist
- Download the dataset using `kagglehub`
- Place all files in the `data/` directory


### Method 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data
2. Download the dataset manually
3. Extract the files into the `data/` directory

## Usage

### 1. Data Exploration

Start with the exploration notebook to understand the dataset:

```bash
jupyter notebook FD001/01_exploration.ipynb
```

### 2. Running Models

Each subdirectory (FD001-FD004) contains notebooks organized by method type:

- **Classical Methods**: `clasic_methods/`
  - Z-score for outlier detection
  - PCA for dimensionality reduction

- **Unsupervised Learning**: `unsupervised_learning/`
  - Isolation Forest
  - One-Class SVM

- **Deep Learning**: `deep_learning/`
  - Basic Autoencoder
  - LSTM Autoencoder
  - TCN-VAE (Temporal Convolutional Network + Variational Autoencoder)


## Results

Results from each experiment are saved in the `outputs/` folders within each method. This includes:
- Detected anomaly plots
- Evaluation metrics
- Trained models (checkpoints)

## Datasets

The project includes experiments with all 4 sub-datasets:

- **FD001**: One operating condition, one failure mode
- **FD002**: Six operating conditions, one failure mode
- **FD003**: One operating condition, two failure modes
- **FD004**: Six operating conditions, two failure modes

## Contributing

To contribute to the project:

1. Create a branch for your feature
2. Implement your changes
3. Make sure notebooks run correctly
4. Create a Pull Request

## License

This project is for academic and research purposes.

## References

- [Dataset on Kaggle](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data)
