# MLOps Project - 23L-2612

A machine learning training pipeline built as part of an MLOps version control assignment, demonstrating Git/GitHub workflows including staging, resets, branching, and merge conflict resolution.

## Project Overview

This project trains a Random Forest Regressor on the California Housing dataset to predict median house values. It is structured to demonstrate a typical MLOps project layout with separated data, source code, and model artifacts.

## Project Structure

mlops-project-23L-2612/
├── data/ # Dataset (gitignored, generated locally)
├── src/
│ └── train_23L-2612.py # Model training script
├── model/ # Trained model output (gitignored)
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md


## Setup

1. Clone the repository:
```bash
   git clone https://github.com/hassan-frq/mlops-project-23L-2612.git
   cd mlops-project-23L-2612
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Generate the dataset:
```bash
   python d.py
```
   This downloads the California Housing dataset and saves it to `data/dataset.csv`.

## Usage

Run the training script:

```bash
python src/train_23L-2612.py
```

This will:
- Load the dataset from `data/dataset.csv`
- Split it into training and test sets
- Train a Random Forest Regressor
- Print the model's R² score on the test set
- Save the trained model to `model/model_23L-2612.pkl`

## Requirements

- Python 3.10+
- pandas
- scikit-learn
- joblib

