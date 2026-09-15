# Kaggle Recommendation Engine

## Dataset Schema

The dataset contains 150,000 interaction records between 5,000 users and
2,000 items. There are no missing values in any of the columns. The data
covers the period from 2022-12-24 to 2024-12-23.

| Column | Data Type | Null Count | Distinct Values | Description | Type |
|---|---|---:|---:|---|---|
| User_ID | object | 0 | 5,000 | Unique identifier of a user. | User |
| Item_ID | object | 0 | 2,000 | Unique identifier of an item. | Item |
| Category | object | 0 | 9 | Category value associated with an interaction. It is not stable for a given item. | Interaction / Context |
| Rating | float64 | 0 | 41 | Explicit rating given in a user-item interaction, ranging from 1.0 to 5.0. | Interaction |
| Timestamp | datetime64[us] | 0 | 731 | Date associated with the interaction. | Interaction |
| Price | float64 | 0 | 47,162 | Price value associated with an interaction. It ranges from 5 to 500 and is not stable for a given item. | Interaction / Context |
| Platform | object | 0 | 4 | Platform associated with an interaction, such as Web or Mobile App. | Interaction / Context |
| Location | object | 0 | 6 | Location value associated with an interaction. It is not stable for a given user. | Interaction / Context |

**Note:** The interpretation of `Category`, `Price`, `Platform`, and `Location`
as interaction/context fields is inferred from the observed data because these
columns do not behave as stable user or item attributes.

### Dataset Summary

**User identifier:** `User_ID`

**Item identifier:** `Item_ID`

**Interaction signal:** `Rating` is an explicit feedback signal. Users provide
ratings between 1.0 and 5.0 rather than the dataset containing only implicit
events such as clicks or views.

**Timestamp:** `Timestamp` records the date of each interaction. The dataset
covers the period from 2022-12-24 to 2024-12-23.

**Dataset size:**
- 5,000 unique users
- 2,000 unique items
- 150,000 interaction records
- 148,892 unique user-item pairs
- 1,108 repeated user-item interactions

## Setup

This project requires Python 3.11 or later.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

On macOS, LightGBM may also require the OpenMP runtime:

```bash
brew install libomp
```

Place the downloaded Kaggle dataset at:

```text
data/raw/personalized_recommendation_dataset.csv
```

The raw dataset is intentionally excluded from Git through `.gitignore`.

## Run Instructions

Start Jupyter Notebook and open the exploration notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/01_exploration.ipynb
```

Run the notebook from top to bottom to reproduce the exploratory analysis, model training, evaluation results, and saved evaluation metrics.

After the final LightGBM model and evaluation metrics have been generated, run the batch recommendation script from the project root:

```bash
python -m src.generate_recommendations
```

This script loads the fitted LightGBM model from disk and generates top-10 recommendations for all training users.

The final output files are:

```text
outputs/recommendations.csv
outputs/metrics.json
```

`recommendations.csv` contains the ranked recommendations with columns:

```text
user_id, item_id, score, rank, generated_at
```

`metrics.json` contains the final evaluation metrics, selected model configuration, and batch runtime.

## Results

The models were evaluated using Precision@10, Recall@10, Hit Rate@10, MAP@10, NDCG@10, catalogue coverage, and average popularity rank.

| Model | Precision@10 | Recall@10 | Hit Rate@10 | MAP@10 | NDCG@10 | Catalogue Coverage | Avg. Popularity Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.001807 | 0.004559 | 0.017861 | 0.001226 | 0.003048 | 1.0000 | 1001.70 |
| Most Popular | 0.001745 | 0.005026 | 0.017445 | 0.001576 | 0.003427 | 0.0065 | 5.69 |
| Item-item kNN | 0.001620 | 0.004148 | 0.015992 | 0.001138 | 0.002759 | 0.9970 | 1028.28 |
| LightGBM 1:10 | 0.001807 | 0.005122 | 0.018069 | 0.001349 | 0.003233 | 0.6255 | 151.94 |

The Most Popular baseline achieved the strongest MAP@10 and NDCG@10, showing that global popularity is a strong signal in this dataset. However, its catalogue coverage is extremely low, meaning that it repeatedly recommends only a very small subset of items.

The LightGBM 1:10 model achieved the best Recall@10 and Hit Rate@10 while covering 62.55% of the catalogue. It was therefore selected as the final model because it provides a better balance between recommendation accuracy and catalogue diversity, even though it does not outperform the Most Popular baseline on every ranking metric.
