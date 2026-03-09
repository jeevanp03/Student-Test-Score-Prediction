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
│   ├── xgboost_baseline.ipynb           # XGBoost baseline model
│   ├── xgboost_improved.ipynb           # XGBoost with tuned hyperparameters
│   ├── neural_network_linear_embedded.ipynb  # Neural network linear embedded model
│   └── ensemble.ipynb                        # Ensemble (XGBoost Improved + NN blend)
├── models/                # Saved model files (.pkl, gitignored)
├── metrics/               # Saved metrics CSVs (gitignored)
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

## Submission

Make sure your Kaggle API credentials are loaded, then submit:

```bash
source .env
kaggle competitions submit \
  -c playground-series-s6e1 \
  -f "submission/neural_network_linear_embedded_submission.csv" \
  -m "Message"
```

Replace the file path with your submission file and `"Message"` with a description of your submission (e.g., model type, changes made).

To check your submission results:

```bash
kaggle competitions submissions -c playground-series-s6e1
```

Or view them on the [competition leaderboard](https://www.kaggle.com/competitions/playground-series-s6e1/leaderboard).

## Architecture

### Neural Network Linear Embedded

```mermaid
flowchart TD
    subgraph EMB["Embeddings"]
        E1["Cat₁<br/>Emb(3→4)"]
        E2["Cat₂<br/>Emb(7→8)"]
        E3["Cat₃<br/>Emb(2→3)"]
        E4["Cat₄<br/>Emb(3→4)"]
        E5["Cat₅<br/>Emb(5→6)"]
        E6["Cat₆<br/>Emb(3→4)"]
        E7["Cat₇<br/>Emb(3→3)"]
        NUM["Numeric<br/>12 features"]
    end

    E1 & E2 & E3 & E4 & E5 & E6 & E7 & NUM --> CAT

    CAT["⊕  Concatenate → 44d"]

    CAT --> L1

    subgraph B1["Dense Block 1"]
        L1["Linear(44 → 256)"]
        BN1["BatchNorm1d(256)"]
        R1["LeakyReLU(α=0.01)"]
        D1["Dropout(p=0.15)"]
        L1 --> BN1 --> R1 --> D1
    end

    D1 --> L2

    subgraph B2["Dense Block 2"]
        L2["Linear(256 → 128)"]
        BN2["BatchNorm1d(128)"]
        R2["LeakyReLU(α=0.01)"]
        D2["Dropout(p=0.15)"]
        L2 --> BN2 --> R2 --> D2
    end

    D2 --> L3

    subgraph B3["Dense Block 3"]
        L3["Linear(128 → 64)"]
        BN3["BatchNorm1d(64)"]
        R3["LeakyReLU(α=0.01)"]
        D3["Dropout(p=0.15)"]
        L3 --> BN3 --> R3 --> D3
    end

    D3 --> OUT["Linear(64 → 1)<br/>Score Prediction"]

    style EMB fill:#0c1a2e,stroke:#0ea5e9,color:#7dd3fc
    style B1  fill:#0c1a2e,stroke:#3b82f6,color:#93c5fd
    style B2  fill:#0c1a2e,stroke:#3b82f6,color:#93c5fd
    style B3  fill:#0c1a2e,stroke:#3b82f6,color:#93c5fd

    style E1  fill:#082030,stroke:#0ea5e9,color:#7dd3fc
    style E2  fill:#082030,stroke:#0ea5e9,color:#7dd3fc
    style E3  fill:#082030,stroke:#0ea5e9,color:#7dd3fc
    style E4  fill:#082030,stroke:#0ea5e9,color:#7dd3fc
    style E5  fill:#082030,stroke:#0ea5e9,color:#7dd3fc
    style E6  fill:#082030,stroke:#0ea5e9,color:#7dd3fc
    style E7  fill:#082030,stroke:#0ea5e9,color:#7dd3fc
    style NUM fill:#0f1f0f,stroke:#4ade80,color:#86efac

    style CAT fill:#1e1030,stroke:#8b5cf6,color:#c4b5fd

    style L1  fill:#0f1a30,stroke:#3b82f6,color:#93c5fd
    style BN1 fill:#1a1200,stroke:#f59e0b,color:#fcd34d
    style R1  fill:#001a10,stroke:#10b981,color:#6ee7b7
    style D1  fill:#1a0a0a,stroke:#ef4444,color:#fca5a5

    style L2  fill:#0f1a30,stroke:#3b82f6,color:#93c5fd
    style BN2 fill:#1a1200,stroke:#f59e0b,color:#fcd34d
    style R2  fill:#001a10,stroke:#10b981,color:#6ee7b7
    style D2  fill:#1a0a0a,stroke:#ef4444,color:#fca5a5

    style L3  fill:#0f1a30,stroke:#3b82f6,color:#93c5fd
    style BN3 fill:#1a1200,stroke:#f59e0b,color:#fcd34d
    style R3  fill:#001a10,stroke:#10b981,color:#6ee7b7
    style D3  fill:#1a0a0a,stroke:#ef4444,color:#fca5a5

    style OUT fill:#1a0c00,stroke:#f97316,color:#fdba74
```

## Results

### Local Evaluation (Validation Set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ensemble (XGBoost + NN) | 6.9681 | 8.7399 | 0.7852 |
| XGBoost Improved | 6.9674 | 8.7423 | 0.7851 |
| Neural Network Linear Embedded | 7.0562 | 8.8355 | 0.7815 |
| Linear Regression | 7.1013| 8.8948 | 0.7789 |
| Genetic Algorithm | 7.0934 | 8.8865 | 0.7780 |
| XGBoost Baseline | 7.0829 | 8.9026 | 0.7771 |
| Random Forest | 7.2497 | 9.1079 | 0.7668 |
| Nearest Neighbour | 7.5878 | 9.4681 | 0.7479 |

### Kaggle Leaderboard (RMSE)

| Model | Private Score | Public Score |
|---|---|---|
| **Ensemble (XGBoost + NN)** | **8.75216** | **8.72378** |
| XGBoost Improved | 8.75240 | 8.72307 |
| Neural Network Linear Embedded | 8.85988 | 8.84211 |
| Linear Regression | 8.89132 | 8.87232 |
| Genetic Algorithm | 8.89189 | 8.87294 |
| XGBoost Baseline | 8.90292 | 8.86689 |
| Random Forest | 9.10425 | 9.07951 |
| Nearest Neighbour | 9.46447 | 9.42413 |

Lower RMSE is better. Public Score is used for competition ranking. The **Ensemble** achieved the best performance on both local validation and the Kaggle leaderboard.
