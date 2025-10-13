#!/usr/bin/env python
# coding: utf-8

"""
Airbnb Dynamic Pricing Optimization

"""
import os
import model
import seaborn as sns  # Correct
from matplotlib import pyplot as plt

# =============================================================================
# SECTION 0: SETUP & INSTALLATION
# =============================================================================


print("AIRBNB DYNAMIC PRICING OPTIMIZATION")


# Import libraries
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

# Others
import joblib
from datetime import datetime
from scipy.stats import randint, uniform

# Set display options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("\n All libraries imported successfully!")

# =============================================================================
# SECTION 1: LOAD THE DATASET
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 1: LOADING DATASET")


# Load the dataset
# Get the current path where the code is saved
path = os.getcwd()

# Read the excel file with data. Make sure it is saved in the same location as the code!!
df = pd.read_excel("Airbnb_Open_Data.xlsx")

# Give a description of the data that was loaded
print("\nHead:\n", df.head())
print("\nDescription:\n", df.price.describe())

# Quick preview
print("\n📋 First 5 rows:")
print(df.head())

print("\n📋 Column Names:")
print(df.columns.tolist())
# =============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")


# 2.1 Dataset Overview
print("\n2.1 Dataset Overview")
print("-" * 40)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

print("\nData Types:")
print(df.dtypes.value_counts())

# 2.2 Missing Data Analysis
print("\n2.2 Missing Data Analysis")
print("-" * 40)

missing_data = df.isnull().sum()
missing_pct = 100 * missing_data / len(df)

missing_df = pd.DataFrame({
    'Missing_Count': missing_data,
    'Percentage': missing_pct
}).sort_values('Percentage', ascending=False)

missing_df = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_df) > 0:
    print(f"\nColumns with missing data: {len(missing_df)}")
    print("\nTop 15 columns with missing values:")
    print(missing_df.head(15))

    # Visualize missing data
    plt.figure(figsize=(12, 6))
    top_missing = missing_df.head(20)
    plt.barh(range(len(top_missing)), top_missing['Percentage'], color='coral')
    plt.yticks(range(len(top_missing)), top_missing.index)
    plt.xlabel('Missing Percentage (%)', fontsize=12)
    plt.title('Top 20 Features with Missing Data', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('01_missing_data.png', dpi=300, bbox_inches='tight')
    print("\n Saved: 01_missing_data.png")
    plt.close()
else:
    print("\n No missing data found!")

# 2.3 Target Variable Analysis (Price)
print("\n2.3 Target Variable Analysis (Price)")
print("-" * 40)

if 'price' in df.columns:
    # Clean price column
    if df['price'].dtype == 'object':
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

    print(f"\nMean Price: ${df['price'].mean():.2f}")
    print(f"Median Price: ${df['price'].median():.2f}")
    print(f"Std Dev: ${df['price'].std():.2f}")
    print(f"Min Price: ${df['price'].min():.2f}")
    print(f"Max Price: ${df['price'].max():.2f}")
    print(f"Skewness: {df['price'].skew():.2f}")
    print(f"Kurtosis: {df['price'].kurtosis():.2f}")

    print("\nPrice Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: ${df['price'].quantile(p / 100):.2f}")

    # Visualize price distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original distribution
    axes[0].hist(df['price'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(df['price'].median(), color='red', linestyle='--', linewidth=2,
                    label=f'Median: ${df["price"].median():.0f}')
    axes[0].axvline(df['price'].mean(), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: ${df["price"].mean():.0f}')
    axes[0].set_xlabel('Price ($)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Price Distribution (Original)', fontsize=13, fontweight='bold')
    axes[0].legend()

    # Log-transformed
    axes[1].hist(np.log1p(df['price'].dropna()), bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1].set_xlabel('Log(Price)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Price Distribution (Log-Transformed)', fontsize=13, fontweight='bold')

    # Box plot
    axes[2].boxplot(df['price'].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[2].set_ylabel('Price ($)', fontsize=11)
    axes[2].set_title('Price Box Plot', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('02_price_distribution.png', dpi=300, bbox_inches='tight')
    print("\n Saved: 02_price_distribution.png")
    plt.close()
else:
    print("\n 'price' column not found in dataset")

# 2.4 Categorical Variables
print("\n2.4 Categorical Variables Analysis")
print("-" * 40)

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nCategorical Features: {len(categorical_cols)}")

if len(categorical_cols) > 0:
    cat_summary = pd.DataFrame({
        'Column': categorical_cols,
        'Unique_Values': [df[col].nunique() for col in categorical_cols],
        'Most_Common': [df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A' for col in categorical_cols]
    }).sort_values('Unique_Values', ascending=False)

    print("\nTop 15 categorical features by cardinality:")
    print(cat_summary.head(15))

# 2.5 Correlation Analysis
print("\n2.5 Correlation Analysis")
print("-" * 40)

if 'price' in df.columns:
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) > 1:
        price_corr = numeric_df.corr()['price'].sort_values(ascending=False)

        print("\nTop 15 Features Correlated with Price:")
        print(price_corr.head(15))

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Correlation bar plot
        top_corr = price_corr.head(15)
        colors = ['green' if x > 0 else 'red' for x in top_corr.values]
        axes[0].barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
        axes[0].set_yticks(range(len(top_corr)))
        axes[0].set_yticklabels(top_corr.index)
        axes[0].set_xlabel('Correlation with Price', fontsize=11)
        axes[0].set_title('Top 15 Price Correlations', fontsize=13, fontweight='bold')
        axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0].invert_yaxis()

        # Heatmap
        top_features = price_corr.head(10).index
        corr_matrix = numeric_df[top_features].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('Correlation Heatmap (Top 10)', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig('03_correlations.png', dpi=300, bbox_inches='tight')
        print("\n Saved: 03_correlations.png")
        plt.close()

# 2.6 Geospatial Analysis
print("\n2.6 Geospatial Analysis")
print("-" * 40)

if all(col in df.columns for col in ['latitude', 'longitude', 'price']):
    sample_size = min(5000, len(df))
    df_sample = df.sample(sample_size, random_state=42)

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(
        df_sample['longitude'],
        df_sample['latitude'],
        c=df_sample['price'],
        cmap='YlOrRd',
        alpha=0.6,
        s=20,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Price ($)')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(f'Geographic Distribution (n={sample_size:,})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('04_geographic_map.png', dpi=300, bbox_inches='tight')
    print(f"\n Saved: 04_geographic_map.png (sample of {sample_size:,} listings)")
    plt.close()
else:
    print("\n Geospatial columns not available")


# =============================================================================
# SECTION 3: DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 3: DATA PREPROCESSING")


# 3.1 Handle Outliers in Price
print("\n3.1 Removing Price Outliers")
print("-" * 40)

if 'price' in df.columns:
    print(f"Original dataset size: {len(df):,}")

    lower_bound = df['price'].quantile(0.01)
    upper_bound = df['price'].quantile(0.99)

    print(f"Price range (1st-99th percentile): ${lower_bound:.2f} - ${upper_bound:.2f}")

    original_len = len(df)
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

    print(f"After outlier removal: {len(df):,} rows")
    print(
        f"Removed: {original_len - len(df):,} outlier listings ({100 * (original_len - len(df)) / original_len:.1f}%)")

# 3.2 Convert Data Types
print("\n3.2 Converting Data Types")
print("-" * 40)

numeric_conversions = ['bedrooms', 'bathrooms', 'beds', 'accommodates',
                       'minimum_nights', 'maximum_nights', 'number_of_reviews',
                       'reviews_per_month', 'calculated_host_listings_count',
                       'availability_365']

converted_count = 0
for col in numeric_conversions:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        converted_count += 1

print(f" Converted {converted_count} columns to numeric type")

# 3.3 Handle Date Columns
print("\n3.3 Processing Date Columns")
print("-" * 40)

date_cols = ['host_since', 'first_review', 'last_review']

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_days_ago'] = (pd.Timestamp.now() - df[col]).dt.days
        print(f" Processed: {col}")

print("\n3.3.1 Dropping Original Datetime Columns")
print("-" * 40)
datetime_cols_to_drop = [col for col in date_cols if col in df.columns]
if datetime_cols_to_drop:
    df = df.drop(columns=datetime_cols_to_drop)
    print(f" Dropped {len(datetime_cols_to_drop)} datetime columns: {datetime_cols_to_drop}")

# 3.4 Handle Boolean Columns
print("\n3.4 Converting Boolean Columns")
print("-" * 40)

bool_cols = ['host_is_superhost', 'host_identity_verified', 'instant_bookable']

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map({'t': 1, 'f': 0, True: 1, False: 0})
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f" Converted: {col}")


# =============================================================================
# SECTION 5: PREPARE DATA FOR MODELING
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 5: PREPARING DATA FOR MODELING")


# Drop unnecessary columns
print("\n5.1 Dropping Unnecessary Columns")
print("-" * 40)

drop_cols = ['id', 'listing_url', 'scrape_id', 'picture_url', 'host_url',
             'host_thumbnail_url', 'host_picture_url', 'name', 'summary',
             'space', 'description', 'neighborhood_overview', 'notes',
             'transit', 'access', 'interaction', 'house_rules', 'host_name',
             'host_about', 'amenities', 'first_review', 'last_review',
             'host_since']

existing_drop = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=existing_drop, errors='ignore')
print(f" Dropped {len(existing_drop)} columns")

# Separate features and target
print("\n5.2 Separating Features and Target")
print("-" * 40)

target_col = 'price'
if target_col not in df.columns:
    print(f" Error: Target column '{target_col}' not found!")
    exit()

y = df[target_col].copy()
X = df.drop(columns=[target_col])

print(f" Features (X): {X.shape}")
print(f" Target (y): {y.shape}")

# Handle categorical variables
print("\n5.3 Encoding Categorical Variables")
print("-" * 40)

categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if X[col].nunique() > 50:
        # High cardinality - target encoding
        target_mean = df.groupby(col)[target_col].mean()
        X[col] = X[col].map(target_mean)
        X[col].fillna(df[target_col].mean(), inplace=True)
        print(f"  Target encoded: {col} ({X[col].nunique()} unique values)")
    else:
        # Low cardinality - one-hot encoding
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=[col])
        print(f"  One-hot encoded: {col}")

# Handle missing values
print("\n5.4 Handling Missing Values")
print("-" * 40)

numeric_cols = X.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    missing_count = X[numeric_cols].isnull().sum().sum()

    if missing_count > 0:
        print(f"Found {missing_count} missing values")

        # Simple median imputation (much faster)
        for col in numeric_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        print(f" Imputed {len(numeric_cols)} columns using median strategy")
    else:
        print(" No missing values found")

# Remove any remaining NaN
X = X.dropna(axis=1, how='all')
valid_idx = ~y.isna()
X = X[valid_idx]
y = y[valid_idx]

print("\n5.4.1 Final Datetime Column Check")
print("-" * 40)
datetime_cols_remaining = X.select_dtypes(include=['datetime64']).columns.tolist()
if datetime_cols_remaining:
    print(f"  Found remaining datetime columns: {datetime_cols_remaining}")
    X = X.drop(columns=datetime_cols_remaining)
    print(f" Dropped {len(datetime_cols_remaining)} datetime columns")
else:
    print(" No datetime columns found - ready for scaling")

print(f"\n Final dataset shape:")
print(f"   X: {X.shape}")
print(f"   y: {y.shape}")

# Split data
print("\n5.5 Splitting Data")
print("-" * 40)

# First split: 70% train, 30% temp (for validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=34)

# Second split: Split the 30% into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=34)


print(f" Train set: {X_train.shape[0]:,} samples (70%)")
print(f" Validation set: {X_val.shape[0]:,} samples (15%)")
print(f" Test set: {X_test.shape[0]:,} samples (15%)")

print("\n5.5.1 Storing Feature Names")
print("-" * 40)
feature_names = X_train.columns.tolist()
print(f" Stored {len(feature_names)} feature names for later use")

# Scale features
print("\n5.6 Scaling Features")
print("-" * 40)

# Scale the features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(" Features scaled using RobustScaler")
print(f" Total features for modeling: {len(feature_names)}")


# =============================================================================
# SECTION 6: MODEL TRAINING
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 6: MODEL TRAINING")



# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    median_ae = np.median(np.abs(y_true - y_pred))
    within_10pct = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100

    print(f"\n{model_name} Performance:")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Median AE: ${median_ae:,.2f}")
    print(f"  Within ±10%: {within_10pct:.1f}%")

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Median_AE': median_ae,
        'Within_10pct': within_10pct
    }


# Store results
models = {}
results = {}

# 6.1 Baseline Models
print("\n6.1 Training Baseline Models")
print("-" * 40)

# Mean baseline
y_pred_mean = np.full(len(y_val), y_train.mean())
results['Mean_Baseline'] = evaluate_model(y_val, y_pred_mean, 'Mean Baseline')

# Linear Regression
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_val_scaled)
models['Linear_Regression'] = lr
results['Linear_Regression'] = evaluate_model(y_val, y_pred_lr, 'Linear Regression')

# Ridge
print("\nTraining Ridge...")
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_val_scaled)
models['Ridge'] = ridge
results['Ridge'] = evaluate_model(y_val, y_pred_ridge, 'Ridge')

# 6.2 Tree-Based Models
print("\n6.2 Training Tree-Based Models")
print("-" * 40)

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_val_scaled)
models['Random_Forest'] = rf
results['Random_Forest'] = evaluate_model(y_val, y_pred_rf, 'Random Forest')

# XGBoost
print("\nTraining XGBoost...")
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_val_scaled)
models['XGBoost'] = xgb
results['XGBoost'] = evaluate_model(y_val, y_pred_xgb, 'XGBoost')

# LightGBM
print("\nTraining LightGBM...")
lgbm = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train_scaled, y_train)
y_pred_lgbm = lgbm.predict(X_val_scaled)
models['LightGBM'] = lgbm
results['LightGBM'] = evaluate_model(y_val, y_pred_lgbm, 'LightGBM')

# CatBoost
print("\nTraining CatBoost...")
cat = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=0
)
cat.fit(X_train_scaled, y_train)
y_pred_cat = cat.predict(X_val_scaled)
models['CatBoost'] = cat
results['CatBoost'] = evaluate_model(y_val, y_pred_cat, 'CatBoost')

# 6.3 Ensemble Model (Stacking)
print("\n6.3 Training Ensemble Model (Stacking)")
print("-" * 40)

base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('lgbm', LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1))
]

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

print("Training Stacking Ensemble...")
stacking.fit(X_train_scaled, y_train)
y_pred_stack = stacking.predict(X_val_scaled)
models['Stacking_Ensemble'] = stacking
results['Stacking_Ensemble'] = evaluate_model(y_val, y_pred_stack, 'Stacking Ensemble')

# =============================================================================
# SECTION 7: MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 7: MODEL COMPARISON")


# Create comparison dataframe
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values('MAE')

print("\n Model Rankings (sorted by MAE):")
print(comparison_df.to_string())

# Find best model
best_model_name = comparison_df.index[0]
best_model = models[best_model_name]
print(f"\n Best Model: {best_model_name}")
print(f"   MAE: ${comparison_df.loc[best_model_name, 'MAE']:,.2f}")
print(f"   R²: {comparison_df.loc[best_model_name, 'R2']:.4f}")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# MAE comparison
axes[0, 0].barh(comparison_df.index, comparison_df['MAE'], color='skyblue')
axes[0, 0].set_xlabel('Mean Absolute Error ($)', fontsize=11)
axes[0, 0].set_title('MAE by Model', fontsize=12, fontweight='bold')
axes[0, 0].invert_yaxis()

# R² comparison
axes[0, 1].barh(comparison_df.index, comparison_df['R2'], color='lightgreen')
axes[0, 1].set_xlabel('R² Score', fontsize=11)
axes[0, 1].set_title('R² by Model', fontsize=12, fontweight='bold')
axes[0, 1].invert_yaxis()

# MAPE comparison
axes[1, 0].barh(comparison_df.index, comparison_df['MAPE'], color='coral')
axes[1, 0].set_xlabel('MAPE (%)', fontsize=11)
axes[1, 0].set_title('MAPE by Model', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()

# Within 10% accuracy
axes[1, 1].barh(comparison_df.index, comparison_df['Within_10pct'], color='plum')
axes[1, 1].set_xlabel('Accuracy within ±10% (%)', fontsize=11)
axes[1, 1].set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('05_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n Saved: 05_model_comparison.png")
plt.close()

# =============================================================================
# SECTION 8: HYPERPARAMETER OPTIMIZATION
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 8: HYPERPARAMETER OPTIMIZATION")


print(f"\nOptimizing {best_model_name}...")

# Define parameter grid for XGBoost (most common best performer)
if best_model_name == 'XGBoost':
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 7)
    }
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
elif best_model_name == 'LightGBM':
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(20, 150),
        'subsample': uniform(0.6, 0.4)
    }
    base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
elif best_model_name == 'Random_Forest':
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
else:
    print(f" Skipping optimization for {best_model_name}")
    param_dist = None
    base_model = None

if param_dist is not None:
    print(f"Running RandomizedSearchCV (20 iterations, 5-fold CV)...")

    search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=20,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train_scaled, y_train)

    print(f"\n Optimization complete!")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV MAE: ${-search.best_score_:.2f}")

    # Evaluate optimized model
    y_pred_optimized = search.best_estimator_.predict(X_val_scaled)
    results[f'{best_model_name}_Optimized'] = evaluate_model(
        y_val, y_pred_optimized, f'{best_model_name} (Optimized)'
    )

    models[f'{best_model_name}_Optimized'] = search.best_estimator_
    best_model = search.best_estimator_

# =============================================================================
# SECTION 9: FINAL TEST SET EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 9: FINAL TEST SET EVALUATION")


print(f"\nEvaluating best model on held-out test set...")
print(f"Model: {best_model_name}")

y_pred_test = best_model.predict(X_test_scaled)
test_results = evaluate_model(y_test, y_pred_test, 'Test Set (Final)')

# Residual analysis
residuals = y_test - y_pred_test

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Test Set Evaluation - Residual Analysis', fontsize=16, fontweight='bold')

# Predicted vs Actual
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=10)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Price ($)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Price ($)', fontsize=11)
axes[0, 0].set_title('Predicted vs Actual Prices', fontsize=12, fontweight='bold')
axes[0, 0].legend()

# Residual plot
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5, s=10)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=11)
axes[0, 1].set_ylabel('Residuals ($)', fontsize=11)
axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')

# Residual distribution
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='lightblue')
axes[1, 0].axvline(residuals.mean(), color='r', linestyle='--', lw=2,
                   label=f'Mean: ${residuals.mean():.2f}')
axes[1, 0].set_xlabel('Residuals ($)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# Q-Q plot
from scipy import stats

stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('06_test_evaluation.png', dpi=300, bbox_inches='tight')
print("\n Saved: 06_test_evaluation.png")
plt.close()

# =============================================================================
# SECTION 10: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 10: FEATURE IMPORTANCE ANALYSIS")


# Get feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\nCalculating feature importance...")

    importances = best_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n Top 20 Most Important Features:")
    print(feature_imp_df.head(20).to_string())

    # Visualize
    plt.figure(figsize=(10, 8))
    top_features = feature_imp_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top 20 Feature Importance - {best_model_name}',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('07_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n Saved: 07_feature_importance.png")
    plt.close()

    # Permutation importance
    print("\nCalculating permutation importance (this may take a few minutes)...")
    perm_importance = permutation_importance(
        best_model, X_val_scaled, y_val,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)

    print("\n Top 20 Features (Permutation Importance):")
    print(perm_imp_df.head(20).to_string())
else:
    print("\n Feature importance not available for this model type")

# =============================================================================
# SECTION 11: SAVE MODEL AND RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 12: SAVING MODEL AND RESULTS")


# Save the best model
model_filename = 'airbnb_pricing_model.pkl'
joblib.dump({
    'model': best_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'model_name': best_model_name,
    'test_results': test_results,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}, model_filename)

print(f"\n Model saved: {model_filename}")

# Save results to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
print(f" Results saved: model_results.csv")

# Save feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_imp_df.to_csv('feature_importance.csv', index=False)
    print(f" Feature importance saved: feature_importance.csv")

# =============================================================================
# SECTION 12: EXAMPLE PREDICTIONS
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 13: MAKING PREDICTIONS")


print("\nExample: Predicting prices for sample listings...")

# Get 5 random samples from test set
sample_indices = np.random.choice(len(X_test_scaled), 5, replace=False)
X_samples = X_test_scaled.iloc[sample_indices]
y_actual = y_test.iloc[sample_indices]

# Predict
y_predicted = best_model.predict(X_samples)

# Display results
print("\n Sample Predictions:")
print("-" * 60)
for i, (actual, pred) in enumerate(zip(y_actual, y_predicted), 1):
    error = actual - pred
    error_pct = (error / actual) * 100
    print(f"Listing {i}:")
    print(f"  Actual Price:    ${actual:,.2f}")
    print(f"  Predicted Price: ${pred:,.2f}")
    print(f"  Error:           ${error:,.2f} ({error_pct:+.1f}%)")
    print()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETE! 🎉")

print("\n Final Model Performance:")
print(f"   Model: {best_model_name}")
print(f"   Test MAE: ${test_results['MAE']:,.2f}")
print(f"   Test RMSE: ${test_results['RMSE']:,.2f}")
print(f"   Test R²: {test_results['R2']:.4f}")
print(f"   Test MAPE: {test_results['MAPE']:.2f}%")
print(f"   Within ±10% Accuracy: {test_results['Within_10pct']:.1f}%")

