# -*- coding: utf-8 -*-
"""Final Project - Combined EDA & Machine Learning Pipeline

SWA2124 Group 27 - Online News Popularity
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Configure global plot aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

# ==============================================================================
# PART 1: DATA LOADING & EDA (SCRIPT 1)
# ==============================================================================

# 1. DATA LOADING
file_paths = [
    "/content/sample_data/OnlineNewsPopularity.csv",
    "/content/OnlineNewsPopularity.csv",
    "OnlineNewsPopularity.csv",
]

df = None
for path in file_paths:
  if os.path.exists(path):
    df = pd.read_csv(path)
    print(f"Dataset successfully loaded from: {path}")
    break

if df is None:
  from google.colab import files

  print("Please upload OnlineNewsPopularity.csv:")
  uploaded = files.upload()
  file_name = list(uploaded.keys())[0]
  df = pd.read_csv(file_name)

print(f"Initial Dataset Shape: {df.shape}")
df.columns = df.columns.str.strip()

# 2. DATA CLEANING & PREPROCESSING
print("\n--- MISSING VALUES CHECK ---")
missing_values = df.isnull().sum().sum()
print(f"Total missing values in dataset: {missing_values}")

print("\n--- DUPLICATES CHECK ---")
duplicates = df.duplicated().sum()
print(f"Total duplicate rows in dataset: {duplicates}")
if duplicates > 0:
  df = df.drop_duplicates()
  print(f"Dropped duplicates. New shape: {df.shape}")

# Remove known scraping artifact (n_unique_tokens ratio > 1)
df_clean = df[df['n_unique_tokens'] <= 1.0].reset_index(drop=True)

# Drop non-predictive metadata columns
df_clean = df_clean.drop(columns=['url', 'timedelta'], errors='ignore')

# Create Target Label (Shares > 1400 = 1, Else = 0)
MEDIAN_SHARES = 1400
df_clean['popular'] = (df_clean['shares'] > MEDIAN_SHARES).astype(int)
df_clean['log_shares'] = np.log1p(df_clean['shares'])

print("\n--- BALANCED TARGET CLASS DISTRIBUTION ---")
print(df_clean['popular'].value_counts(normalize=True))


# 3. EXPLORATORY DATA ANALYSIS (EDA) PLOTS

# Chart 1: Popular vs Not Popular Count
plt.figure(figsize=(6, 4))
ax = sns.countplot(
    x='popular',
    hue='popular',
    data=df_clean,
    palette=['#e74c3c', '#2ecc71'],
    legend=False,
)
plt.title(
    'Target Class Distribution (Popular vs Not Popular)',
    fontsize=11,
    fontweight='bold',
)
plt.xlabel('Class (0: Not Popular, 1: Popular)', fontsize=10)
plt.ylabel('Article Count', fontsize=10)
plt.xticks([0, 1], ['Not Popular (<=1,400)', 'Popular (>1,400)'])
for p in ax.patches:
  ax.annotate(
      f'{int(p.get_height()):,}',
      (p.get_x() + p.get_width() / 2.0, p.get_height() / 2),
      ha='center',
      va='center',
      color='white',
      fontweight='bold',
      fontsize=11,
  )
plt.tight_layout()
plt.savefig('chart1_target_distribution.png', bbox_inches='tight')
plt.show()

# Chart 2: Shares Distribution (Skewed vs Log-Transformed)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df_clean['shares'], bins=50, kde=True, ax=axes[0], color='#3498db')
axes[0].set_title('Raw Shares Distribution', fontweight='bold')
axes[0].set_xlabel('Shares')
axes[0].set_xlim(0, 20000)

sns.histplot(
    df_clean['log_shares'], bins=50, kde=True, ax=axes[1], color='#9b59b6'
)
axes[1].set_title('Log-Transformed Shares Distribution', fontweight='bold')
axes[1].set_xlabel('log(shares)')
plt.tight_layout()
plt.savefig('chart2_shares_distribution.png', bbox_inches='tight')
plt.show()

# Chart 3: Average Shares by Article Category
channels = [
    'data_channel_is_lifestyle',
    'data_channel_is_entertainment',
    'data_channel_is_bus',
    'data_channel_is_socmed',
    'data_channel_is_tech',
    'data_channel_is_world',
]
channel_labels = [
    'Lifestyle',
    'Entertainment',
    'Business',
    'Social Media',
    'Tech',
    'World',
]
channel_popular_rates = [
    df_clean[df_clean[ch] == 1]['popular'].mean() for ch in channels
]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(channel_labels, channel_popular_rates, color='#34495e')
plt.axhline(0.5, color='red', linestyle='--', label='Dataset Average (50%)')
plt.title(
    'Popularity Rate by News Article Category', fontsize=11, fontweight='bold'
)
plt.ylabel('Proportion of Popular Articles', fontsize=10)
plt.ylim(0, 0.85)
for bar in bars:
  yval = bar.get_height()
  plt.text(
      bar.get_x() + bar.get_width() / 2,
      yval + 0.015,
      f'{yval:.1%}',
      ha='center',
      va='bottom',
      fontweight='bold',
      fontsize=10,
  )
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('chart3_category_popularity.png', bbox_inches='tight')
plt.show()

# Chart 4: Average Shares by Weekday/Weekend
days = [
    'weekday_is_monday',
    'weekday_is_tuesday',
    'weekday_is_wednesday',
    'weekday_is_thursday',
    'weekday_is_friday',
    'weekday_is_saturday',
    'weekday_is_sunday',
]
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_rates = [df_clean[df_clean[day] == 1]['popular'].mean() for day in days]
colors = ['#7f8c8d'] * 5 + ['#e67e22'] * 2

plt.figure(figsize=(8, 4.5))
bars = plt.bar(day_labels, day_rates, color=colors)
plt.axhline(0.5, color='red', linestyle='--', alpha=0.7)
plt.title(
    'Article Popularity Rate by Day of Publication',
    fontsize=11,
    fontweight='bold',
)
plt.ylabel('Popularity Rate', fontsize=10)
plt.ylim(0, 0.75)
for bar in bars:
  yval = bar.get_height()
  plt.text(
      bar.get_x() + bar.get_width() / 2,
      yval + 0.015,
      f'{yval:.1%}',
      ha='center',
      va='bottom',
      fontweight='bold',
      fontsize=10,
  )
plt.tight_layout()
plt.savefig('chart4_weekday_weekend_popularity.png', bbox_inches='tight')
plt.show()

# Chart 5: Correlation Heatmap
plt.figure(figsize=(9, 7))
numeric_df = df_clean.select_dtypes(include=[np.number])
candidates = numeric_df.drop(
    columns=['popular', 'shares', 'log_shares'], errors='ignore'
)
top_corr_features = (
    candidates.corrwith(df_clean['popular'])
    .abs()
    .sort_values(ascending=False)
    .head(10)
    .index
)
corr_matrix = df_clean[list(top_corr_features) + ['popular']].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    vmin=-1,
    vmax=1,
)
plt.title(
    'Correlation Heatmap of Top 10 Predictive Features',
    fontsize=11,
    fontweight='bold',
)
plt.tight_layout()
plt.savefig('chart5_correlation_heatmap.png', bbox_inches='tight')
plt.show()


# ==============================================================================
# PART 2: MACHINE LEARNING PIPELINE (SCRIPT 2)
# ==============================================================================

# Note: The ML pipeline re-loads the dataset to maintain strict independent execution
df_ml = pd.read_csv('OnlineNewsPopularity.csv')
df_ml.columns = df_ml.columns.str.strip()  # dataset has leading spaces in headers
df_ml = df_ml.drop(
    columns=['url', 'timedelta']
)  # non-predictive identifier columns

THRESHOLD = 1400  # median shares in this dataset
df_ml['Popular'] = (df_ml['shares'] >= THRESHOLD).astype(int)
df_ml = df_ml.drop(columns=['shares'])  # drop raw target, keep only the label

print(f'\nDataset shape for ML Pipeline: {df_ml.shape}')
print(df_ml['Popular'].value_counts(normalize=True).rename('proportion'))

# 1. Train-Test Split
X = df_ml.drop(columns=['Popular'])
y = df_ml['Popular']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Training set: {X_train.shape}, Test set: {X_test.shape}')
print('\nClass balance (train):')
print(y_train.value_counts(normalize=True).rename('proportion'))
print('\nClass balance (test):')
print(y_test.value_counts(normalize=True).rename('proportion'))

# 2. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    X_test_scaled, columns=X_test.columns, index=X_test.index
)

print('\nScaled feature means (should all be ~0):')
print(X_train_scaled.mean().round(3).head())
print('\nScaled feature std devs (should all be ~1):')
print(X_train_scaled.std().round(3).head())

# 3. Train & Evaluate 5 Classification Models
models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=1000, random_state=42),
        True,
    ),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), False),
    'Random Forest': (
        RandomForestClassifier(n_estimators=200, random_state=42),
        False,
    ),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), False),
    'KNN': (KNeighborsClassifier(n_neighbors=15), True),
}

results = []
confusion_matrices = {}
fitted_models = {}  # Save fitted models for later analysis

for name, (model, needs_scaling) in models.items():
  Xtr = X_train_scaled if needs_scaling else X_train
  Xte = X_test_scaled if needs_scaling else X_test

  model.fit(Xtr, y_train)
  y_pred = model.predict(Xte)

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  results.append({
      'Model': name,
      'Accuracy': round(acc, 3),
      'Precision': round(prec, 3),
      'Recall': round(rec, 3),
      'F1-Score': round(f1, 3),
  })
  confusion_matrices[name] = cm
  fitted_models[name] = model  # Store fitted model instance

  print(f"\n{'='*50}\n{name}\n{'='*50}")
  print(
      classification_report(
          y_test, y_pred, target_names=['Not Popular', 'Popular']
      )
  )

# 4. Model Comparison Table
results_df = (
    pd.DataFrame(results)
    .sort_values('F1-Score', ascending=False)
    .reset_index(drop=True)
)
print('\n--- MODEL COMPARISON TABLE ---')
print(results_df.to_string(index=False))

results_df.to_csv('member4_model_comparison_results.csv', index=False)

# 5. Confusion Matrix Visualizations
labels = ['Not Popular', 'Popular']
f1s = {row['Model']: row['F1-Score'] for row in results}
best_model = max(f1s, key=f1s.get)

# Single large confusion matrix - best model
fig, ax = plt.subplots(figsize=(5, 4.2))
sns.heatmap(
    confusion_matrices[best_model],
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    cbar=False,
    annot_kws={'size': 14},
    ax=ax,
)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('Actual Label', fontsize=11)
ax.set_title(f'Confusion Matrix - {best_model}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_best_model.png', dpi=200, bbox_inches='tight')
plt.show()

print(f'\nBest model: {best_model} (F1 = {f1s[best_model]:.3f})')

# Grid of all 5 confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for i, (name, cm) in enumerate(confusion_matrices.items()):
  sns.heatmap(
      cm,
      annot=True,
      fmt='d',
      cmap='Blues',
      xticklabels=labels,
      yticklabels=labels,
      cbar=False,
      annot_kws={'size': 11},
      ax=axes[i],
  )
  axes[i].set_title(f'{name}\n(F1 = {f1s[name]:.3f})', fontsize=11)
  axes[i].set_xlabel('Predicted', fontsize=9)
  axes[i].set_ylabel('Actual', fontsize=9)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('confusion_matrix_all_models.png', dpi=200, bbox_inches='tight')
plt.show()

# 6. Feature Importance (Gradient Boosting)
gb_model = fitted_models['Gradient Boosting']
gb_importances = gb_model.feature_importances_
feature_names = X_train.columns

gb_importance_df = pd.DataFrame(
    {'Feature': feature_names, 'Importance': gb_importances}
)

selected_features = [
    'kw_avg_avg',
    'self_reference_avg_shares',
    'is_weekend',
    'data_channel_is_tech',
    'n_tokens_content',
    'title_sentiment_polarity',
]

gb_selected_df = gb_importance_df[
    gb_importance_df['Feature'].isin(selected_features)
].sort_values(by='Importance', ascending=False)

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    x='Importance',
    y='Feature',
    data=gb_selected_df,
    hue='Feature',
    palette='viridis',
    legend=False,
)

for p in ax.patches:
  width = p.get_width()
  ax.annotate(
      f'{width:.3f}',
      (width + 0.002, p.get_y() + p.get_height() / 2.0),
      ha='left',
      va='center',
      fontsize=10,
  )

plt.title(
    'Top Predictive Features Influencing Article Popularity',
    fontsize=13,
    fontweight='bold',
)
plt.xlabel('Relative Importance Score', fontsize=11)
plt.ylabel('Feature Name', fontsize=11)
plt.xlim(0, max(gb_selected_df['Importance']) * 1.15)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('gb_feature_importance_selected.png', dpi=300)
plt.show()