# Student Test Score Prediction

MSE 546 Final Project — Group 3

Cecilia Su, Jeevan Parmar, Levenet Eren, Joseph Ngai, Patrick Bennett

Regression model to predict student exam scores based on academic behavior, study habits, lifestyle routines, and exam conditions. Built for the Kaggle [Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1) competition.

## Dataset

The dataset contains 20,000 rows with the following features:

| Feature | Type |
|---|---|
| `age` | Quantitative |
| `gender` | Categorical |
| `course` | Categorical |
| `study_hours` | Quantitative |
| `class_attendance` | Quantitative |
| `internet_access` | Binary |
| `sleep_hours` | Quantitative |
| `sleep_quality` | Categorical |
| `study_method` | Categorical |
| `facility_rating` | Categorical |
| `exam_difficulty` | Categorical |
| **`exam_score`** | **Target** |

## Project Structure

```
Student-Test-Score-Prediction/
├── data/                  # CSV files (gitignored, download from Kaggle)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── initial_linear_regression.ipynb  # Baseline linear regression + EDA
│   ├── genetic_algorithm.ipynb          # GA feature selection on linear regression
│   ├── random_forest.ipynb              # Random forest regressor
│   └── neural_network.ipynb             # Neural network model
├── submission/            # Generated submission CSVs (gitignored)
├── .env.example           # Template for API credentials
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/Student-Test-Score-Prediction.git
cd Student-Test-Score-Prediction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) and click **"Create New Token"**
2. Copy the token
3. Create a `.env` file from the example:

```bash
cp .env.example .env
```

4. Open `.env` and paste your token

### 5. Download the data

```bash
source .env
kaggle competitions download -c playground-series-s6e1
unzip playground-series-s6e1.zip -d data/
rm playground-series-s6e1.zip
```

Or download the files manually from [Kaggle](https://www.kaggle.com/competitions/playground-series-s6e1/data) and place them in `data/`.

## Updating Dependencies

After installing a new package, update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

## Evaluation

The primary evaluation metrics are:

- **RMSE** (Root Mean Squared Error) — lower is better
- **R-Squared** — higher is better
