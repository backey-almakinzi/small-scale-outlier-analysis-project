# Outlier Analysis: Cancer Severity Classification

A machine learning project investigating whether class-wise outlier removal improves cancer severity classification.

The project compares four classifiers trained on all training rows against the same classifiers trained on an IQR-filtered subset. Both experiments are evaluated on the **same untouched test set**.

In this experiment, outlier removal slightly reduced performance across all four models. **Logistic Regression trained on all training rows achieved the best results, with 99.58% accuracy and 98.94% macro F1.**

## Dataset

This project uses the [Global Cancer Patients (2015–2024) dataset](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024), available on Kaggle.

The recorded analysis includes 50,000 rows:

| Partition | Rows |
|---|---:|
| Original dataset | 50,000 |
| Training set | 40,000 |
| Test set | 10,000 |
| Training set after IQR filtering | 38,964 |

### Target Classes

The original `Target_Severity_Score` is converted into three categories:

| Severity | Label | Score Range |
|---|---:|---|
| Low | 0 | Score ≤ 3 |
| Medium | 1 | 3 < score ≤ 6 |
| High | 2 | Score > 6 |

These thresholds are defined for this project and are not validated clinical categories.

### Input Features

**Numeric features:**

- Age
- Genetic_Risk
- Air_Pollution
- Alcohol_Use
- Smoking
- Obesity_Level
- Treatment_Cost_USD

**Categorical features:**

- Gender
- Country_Region
- Cancer_Type
- Cancer_Stage

Patient IDs and the original severity score are excluded from the model inputs.

## Methodology

### 1. Data Loading and Validation

The notebook:

- Loads the CSV from a local path or downloads it using KaggleHub.
- Checks that all required columns are present.
- Validates numeric columns.
- Treats infinite numeric values as missing.
- Removes rows with missing target scores.
- Displays a preview and missing-value counts.

Missing predictor values are handled later inside the model pipelines.

### 2. Train/Test Split

A stratified split divides the dataset into:

- **80% training:** 40,000 rows
- **20% testing:** 10,000 rows

The split uses `random_state=42`.

The split is created **before** outlier filtering or preprocessing. Both experiments use the same test rows, and the test set remains unfiltered.

### 3. Class-Wise IQR Filtering

Outlier thresholds are calculated separately for each severity class using only the training data.

For each selected numeric feature:

```text
IQR = Q3 − Q1

Lower bound = Q1 − 1.2 × IQR
Upper bound = Q3 + 1.2 × IQR
```

A training row is removed if any applicable numeric feature falls outside these bounds.

Features with zero IQR do not trigger removal. Missing values do not trigger removal and are handled through imputation.

The filtering process removed **1,036 training rows**, representing **2.59%** of the training set.

This is a supervised training-data selection method because it uses training labels to define groups. It does not require the true severity label when making predictions on new records.

### 4. Preprocessing

Each model and training condition receives an independent preprocessing pipeline.

| Feature Type | Preprocessing |
|---|---|
| Numeric | Median imputation |
| Numeric for Logistic Regression | Median imputation followed by standard scaling |
| Categorical | Most-frequent imputation followed by one-hot encoding |

The categorical encoder ignores unseen test categories.

All preprocessing steps are fitted only on the training rows used by the corresponding experiment.

### 5. Classification Models

| Model | Main Configuration |
|---|---|
| Random Forest | 80 estimators, maximum depth of 8 |
| Logistic Regression | Maximum 2,000 iterations, standardized numeric features |
| Decision Tree | No maximum depth specified |
| Gradient Boosting | 50 estimators, subsample of 0.8 |

A majority-class dummy classifier is also included as a reference baseline.

## Evaluation

The models are evaluated using:

- Accuracy
- Balanced accuracy
- Macro F1-score
- Macro one-versus-rest ROC-AUC
- Per-class classification reports
- Confusion matrices
- One-versus-rest ROC curves

Balanced accuracy averages recall across classes. Macro F1 gives equal weight to each class, making both useful for this imbalanced dataset.

ROC-AUC is calculated using predicted probabilities.

The test set contains:

| Severity | Test Rows |
|---|---:|
| Low | 529 |
| Medium | 7,507 |
| High | 1,964 |

## Results

All results below were obtained using the same untouched 10,000-row test set.

| Model | Training Condition | Accuracy | Balanced Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | All rows | 0.8289 | 0.4676 | 0.4924 | 0.9744 |
| Random Forest | IQR-filtered | 0.8142 | 0.4411 | 0.4595 | 0.9725 |
| Logistic Regression | All rows | **0.9958** | **0.9837** | **0.9894** | 1.0000 |
| Logistic Regression | IQR-filtered | 0.9941 | 0.9732 | 0.9845 | 1.0000 |
| Decision Tree | All rows | 0.8956 | 0.8015 | 0.8169 | 0.8531 |
| Decision Tree | IQR-filtered | 0.8881 | 0.7688 | 0.7958 | 0.8293 |
| Gradient Boosting | All rows | 0.9095 | 0.7040 | 0.7811 | 0.9889 |
| Gradient Boosting | IQR-filtered | 0.9048 | 0.6960 | 0.7736 | 0.9805 |

### Accuracy Change After Filtering

Changes represent filtered-training accuracy minus all-training accuracy.

| Model | Accuracy Change |
|---|---:|
| Random Forest | −1.47 percentage points |
| Logistic Regression | −0.17 percentage points |
| Decision Tree | −0.75 percentage points |
| Gradient Boosting | −0.47 percentage points |

### Findings

- **IQR filtering did not improve performance in this experiment.** Accuracy, balanced accuracy, and macro F1 decreased for every classifier.
- **Logistic Regression performed best**, achieving 99.58% accuracy and 98.94% macro F1 when trained on all training rows.
- Random Forest achieved 82.89% accuracy without filtering, but its balanced accuracy was only 46.76%, indicating weak performance across the minority classes.
- Logistic Regression's ROC-AUC values both round to 1.0000. This does not imply perfect classification or exact equality between the unrounded values.
- The results demonstrate that removing unusual training examples does not necessarily improve classification.

These results replace the earlier comparison, which calculated outlier thresholds before splitting and used different test sets. The corrected evaluation does not support the earlier claim that outlier removal improved performance.

## Visualizations

The notebook includes:

- Training-class distribution
- Class-wise training-row retention counts
- Model comparison charts
- Confusion matrices
- One-versus-rest ROC curves
- Tables showing performance differences after filtering

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- KaggleHub
- Jupyter Notebook

## Installation and Usage

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
```

### 2. Prepare the Dataset

Download the [Global Cancer Patients (2015–2024) dataset](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024).

Place `global_cancer_patients_2015_2024.csv` beside the notebook or inside a `data/` folder.

You can also provide an explicit path in the loading cell:

```python
DATA_PATH = Path("/your/path/to/global_cancer_patients_2015_2024.csv")
```

If no local file or explicit path is provided, the notebook downloads the dataset using KaggleHub.

Downloading requires network access and may require Kaggle authentication. The dataset is not included in the repository.

### 3. Run the Notebook

```bash
jupyter notebook
```

Open `outlier_analysis_project.ipynb` and execute the cells in order.

The notebook will:

1. Load and validate the dataset.
2. Convert severity scores into three classes.
3. Create the stratified train/test split.
4. Calculate training-only IQR thresholds.
5. Create the filtered training subset.
6. Train and evaluate all four classifiers under both conditions.
7. Display comparison tables and visualizations.

Models and metric tables are not automatically exported.

For reproducibility, record the dataset version and installed package versions. The notebook prints the Python and scikit-learn versions.

## Limitations

- The results describe one fixed holdout split and do not establish statistical significance.
- IQR filtering identifies unusual feature values, not confirmed errors. Valid examples, including minority-class examples, may be removed.
- Logistic Regression's near-perfect performance warrants investigating how the original severity score was constructed and whether it was directly derived from the predictors. This is an investigation point, not confirmed evidence of leakage.
- Treatment cost and cancer stage may be unavailable at the intended prediction time. Their timing and relationship to the target should be assessed before making practical predictive claims.
- Dataset provenance, duplicates, and repeated patients require further review. The current split assumes independent rows.
- Further tuning should use validation data or cross-validation within the training partition.
- The project is an educational analysis and is not a validated clinical prediction system.

## Conclusion

For this dataset and evaluation split, retaining all training rows produced better results than class-wise IQR filtering.

Logistic Regression achieved the highest accuracy and macro F1. The experiment highlights the importance of comparing preprocessing methods on a consistent test set and evaluating minority-class performance alongside overall accuracy.
