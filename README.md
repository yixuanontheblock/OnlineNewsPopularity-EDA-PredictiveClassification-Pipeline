# Online News Popularity: EDA & Predictive Classification Pipeline

An end-to-end data science project predicting whether an article will become popular (shares $\ge$ 1,400) based on content, publication timing, keywords, and sentiment features. Developed for **SWA2124 (Group 27)**.

---

## Project Overview

* **Objective:** Clean, visualize, and classify articles from the Online News Popularity dataset into binary popularity categories (`0: Not Popular`, `1: Popular`).
* **Target Metric:** Articles with $\ge$ 1,400 shares (the dataset median threshold) are labeled as popular.
* **Models Evaluated:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and K-Nearest Neighbors (KNN).

---

## Key Findings & EDA Insights

* **Target Balance:** The threshold of 1,400 shares establishes a balanced binary distribution.
* **Timing Effect:** Publication day significantly alters popularity rates, with weekend publications showing higher average engagement compared to mid-week articles.
* **Key Drivers:** Feature importance analysis via Gradient Boosting highlights `kw_avg_avg`, `self_reference_avg_shares`, `is_weekend`, and channel categories as key predictors of article reach.

---

## Pipeline Workflow

1. **Preprocessing & Cleaning**
   * Drops non-predictive metadata columns (`url`, `timedelta`).
   * Removes duplicate entries and filters out web scraping artifacts (`n_unique_tokens > 1.0`).
   * Constructs the binary target variable based on the share threshold.

2. **Exploratory Data Analysis**
   * Target balance count plot (`chart1_target_distribution.png`).
   * Raw vs. log-transformed shares distribution (`chart2_shares_distribution.png`).
   * Proportion of popular articles across data channels (`chart3_category_popularity.png`).
   * Weekday vs. weekend popularity comparison (`chart4_weekday_weekend_popularity.png`).
   * Heatmap showing top correlated features (`chart5_correlation_heatmap.png`).

3. **Model Training & Evaluation**
   * Stratified 80/20 train-test split.
   * Standard scaling applied to feature-distance models (Logistic Regression, KNN).
   * Model evaluation on Accuracy, Precision, Recall, and F1-Score.
   * Export of `member4_model_comparison_results.csv`, confusion matrix heatmaps, and Gradient Boosting feature importance plots.

---

## Installation & Setup

### Requirements

* Python 3.8+
* Dependencies:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn
  ```

### Dataset

Place `OnlineNewsPopularity.csv` in the root directory alongside `final_code.py` (or run directly in Google Colab, where interactive upload is supported).

---

## Running the Pipeline

Execute the script via terminal:

```bash
python final_code.py
```

### Generated Outputs

* **Visualizations:**
  * `chart1_target_distribution.png`
  * `chart2_shares_distribution.png`
  * `chart3_category_popularity.png`
  * `chart4_weekday_weekend_popularity.png`
  * `chart5_correlation_heatmap.png`
  * `confusion_matrix_best_model.png` & `confusion_matrix_all_models.png`
  * `gb_feature_importance_selected.png`
* **Data Artifacts:**
  * `member4_model_comparison_results.csv` (complete metric breakdown per model)
