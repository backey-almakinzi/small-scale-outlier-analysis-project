# Cancer Severity Classification: Impact of Outlier Removal

This project investigates how **outlier removal affects multiclass classification** on a global cancer patient dataset covering **2015–2024**. Four machine-learning classifiers are trained and evaluated using the original data and a version filtered for outliers.

## Project objectives

- Convert a continuous cancer severity score into three classes: **Low**, **Medium**, and **High**.
- Identify and remove outliers using a **class-wise interquartile range (IQR)** approach.
- Compare four classifiers before and after outlier removal.
- Examine accuracy, per-class precision/recall/F1, confusion matrices, and one-vs-rest ROC curves.

## Dataset

**Source:** [Global Cancer Patients (2015–2024) — Kaggle](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024)  
**Dataset identifier used in the notebook:** `zahidmughal2343/global-cancer-patients-2015-2024`  
**CSV file:** `global_cancer_patients_2015_2024.csv`

The notebook's loaded data contains **50,000 records**. It uses demographic, risk-related, disease, and treatment-cost features to predict the `Target_Severity_Score` category.

### Severity classes

| Encoded label | Class | Original severity score |
|---|---|---|
| `0` | Low | Score ≤ 3 |
| `1` | Medium | 3 < Score ≤ 6 |
| `2` | High | Score > 6 |

Class distribution recorded in the notebook before outlier removal:

| Class | Number of records |
|---|---:|
| Low | 2,643 |
| Medium | 37,536 |
| High | 9,821 |

**Note:** The classes are imbalanced, with Medium accounting for most samples. These labels are derived from the dataset's severity score; the notebook does not independently validate clinical severity.

## Methodology

### 1. Load and prepare the data

The dataset is downloaded through `kagglehub` and loaded with pandas. The continuous `Target_Severity_Score` column is mapped to three severity categories using the thresholds above.

The notebook selects the following predictors:

- **Numerical:** `Age`, `Genetic_Risk`, `Air_Pollution`, `Alcohol_Use`, `Smoking`, `Obesity_Level`, `Treatment_Cost_USD`
- **Categorical:** `Gender`, `Country_Region`, `Cancer_Type`, `Cancer_Stage`

Numerical features are passed through unchanged. Categorical features are transformed with `OneHotEncoder(handle_unknown='ignore')` using a scikit-learn `ColumnTransformer`.

### 2. Detect and remove outliers

The notebook applies an **IQR filter separately within each severity class**, iterating over the numeric columns in the dataframe other than the target. A value is flagged when it falls outside:

```text
Lower bound = Q1 − 1.2 × IQR
Upper bound = Q3 + 1.2 × IQR
IQR = Q3 − Q1
```

A row is removed if it is flagged in any numeric feature. The recorded run removes **1,280 records**, leaving **48,720**.

### 3. Train and evaluate classifiers

The notebook trains each classifier on both the original and filtered datasets, using an **80/20 stratified train/test split** with `random_state=42`.

| Model | Notebook configuration |
|---|---|
| Random Forest | `n_estimators=80`, `max_depth=8`, `random_state=42` |
| Logistic Regression | `max_iter=100`, `random_state=42` |
| Decision Tree | `max_depth=None`, `random_state=42` |
| Gradient Boosting | `n_estimators=50`, `subsample=0.8`, `random_state=42` |

Each model runs inside a scikit-learn pipeline with the same categorical encoding step. Evaluation includes **accuracy, classification report, per-class and macro-average ROC curves, and confusion matrix**.

## Results

The following accuracy scores are from the **saved notebook outputs**:

| Model | With outliers | Without outliers | Change |
|---|---:|---:|---:|
| Random Forest | 82.9% | 84.1% | +1.2 percentage points |
| Logistic Regression | 79.3% | 80.9% | +1.6 percentage points |
| Decision Tree | 89.6% | 91.5% | +1.9 percentage points |
| Gradient Boosting | 90.9% | **92.4%** | +1.5 percentage points |

**Gradient Boosting achieved the highest recorded accuracy (92.4%)** on the filtered dataset. Accuracy improved for all four models in the recorded comparison, but these improvements **should not be interpreted as proof that outlier removal improves generalization**: filtering is performed before the train/test split and depends on the true severity labels. The two experiments also evaluate on different test subsets.

Moreover, accuracy alone can obscure performance on the underrepresented Low and High classes. Review the classification reports and confusion matrices for class-specific results.

## Installation and usage

### 1. Get the project

Clone the GitHub repository or download the notebook:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Run the notebook

```bash
jupyter notebook outlier_analysis_project.ipynb
```

Run the cells in order. The notebook downloads the dataset using:

```python
import kagglehub
path = kagglehub.dataset_download("zahidmughal2343/global-cancer-patients-2015-2024")
```

**Important path note:** The current notebook reads the CSV from a hardcoded Kaggle environment location:

```python
df = pd.read_csv('/kaggle/input/global-cancer-patients-2015-2024/global_cancer_patients_2015_2024.csv')
```

That path works when the file is available there (for example, in the corresponding Kaggle notebook environment). For a local Jupyter environment, replace the read step with:

```python
from pathlib import Path
csv_path = Path(path) / "global_cancer_patients_2015_2024.csv"
df = pd.read_csv(csv_path)
```

The notebook will display class counts, a severity distribution plot, classification metrics, ROC curves, and confusion matrices. It does not currently export trained models or a separate evaluation report.

## Limitations and possible improvements

- **Evaluation leakage:** Outliers are filtered using class labels before the train/test split. For a more reliable comparison, split first, fit outlier criteria on training data only, and keep a single untouched test set for both experiments. Label-dependent filtering would also not be directly usable for unlabeled inference data.
- **Class imbalance:** The Medium class dominates. Consider comparing macro-F1, balanced accuracy, and per-class recall alongside accuracy.
- **Logistic Regression convergence:** The notebook records convergence warnings with `max_iter=100`. Numerical scaling and a higher iteration limit may help.
- **Reproducibility:** Record fixed dependency versions and verify the results after rerunning the entire notebook in a clean environment.
- **Interpretation:** This is a machine-learning comparison on a Kaggle dataset, **not a clinically validated cancer severity prediction tool**.

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, KaggleHub, and Jupyter Notebook.

## Dataset credit

Dataset: [Global Cancer Patients (2015–2024)](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024), published on Kaggle by `zahidmughal2343`.
