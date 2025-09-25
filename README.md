# Delivery Time Prediction Models - Master's Thesis Project

## Overview

This project implements and evaluates machine learning models for predicting delivery times in last-mile delivery systems. The study focuses on two Chinese cities (Jilin and Yantai) and compares six different machine learning approaches: Linear Regression, K-Nearest Neighbors, Random Forest, XGBoost, CatBoost, and Transformer-based neural networks.

## Project Structure

```
Master's Thesis/
├── src/                          # Source code modules
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── loaders.py            # Data loading and feature engineering
│   ├── eda/                      # Exploratory Data Analysis
│   │   ├── __init__.py
│   │   ├── plots.py              # EDA visualization functions
│   │   └── quality.py            # Data quality assessment
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py        # Feature creation and weather data
│   └── models/                   # Machine learning models
│       ├── __init__.py
│       ├── classical.py          # Classical ML models
│       └── transformers.py       # Neural network models
├── data/                         # Raw data files
│   ├── delivery_jl.csv           # Jilin city delivery data
│   └── delivery_yt.csv           # Yantai city delivery data
├── analysis/                     # Analysis scripts
│   ├── train.py                  # Main training pipeline
│   ├── ata_eda.py                # Comprehensive EDA toolkit
│   ├── error_analysis.py         # Basic error analysis
│   ├── detailed_error_analysis.py # Advanced error analysis
│   ├── residual_analysis.py      # Residual analysis and diagnostics
│   └── additional_analysis.py    # KNN scatter plots and validation curves
├── results/                      # Analysis outputs
│   ├── error_analysis/           # Basic error analysis results
│   ├── detailed_analysis/        # Advanced error analysis results
│   ├── residual_analysis/        # Residual analysis results
│   ├── additional_analysis/      # KNN and Transformer analysis
│   ├── thesis_figures/           # Publication-quality figures
│   ├── thesis_summary/           # Final summary plots
│   ├── eda_Jilin_features/       # EDA plots for Jilin
│   ├── eda_Yantai_features/      # EDA plots for Yantai
│   ├── ata_eda_Jilin/            # ATA EDA plots for Jilin
│   └── ata_eda_Yantai/           # ATA EDA plots for Yantai
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Conceptual Pipeline Flow

### 1. Data Pipeline
```
Raw Data (CSV) → Data Loading → Data Cleaning → Feature Engineering → Weather Augmentation
```

**Components:**
- **Data Loading**: Load delivery data for Jilin and Yantai cities
- **Data Cleaning**: Remove duplicates, handle missing values, validate coordinates
- **Feature Engineering**: Create temporal features, distance calculations, categorical encodings
- **Weather Augmentation**: Fetch and integrate weather data from Open-Meteo API

### 2. Machine Learning Pipeline
```
Features → Train/Test Split → Model Training → Model Evaluation → Performance Metrics
```

**Models Evaluated:**
1. **Linear Regression**: Baseline linear model
2. **K-Nearest Neighbors**: Instance-based learning
3. **Random Forest**: Ensemble of decision trees
4. **XGBoost**: Gradient boosting framework
5. **CatBoost**: Categorical boosting
6. **Transformer**: Neural network with attention mechanism

### 3. Analysis Pipeline
```
Model Results → Error Analysis → Residual Analysis → Statistical Validation → Thesis Reports
```

**Analysis Components:**
- **Error Analysis**: MAE, RMSE, R², Accuracy metrics
- **Residual Analysis**: Normality tests, heteroscedasticity, outlier detection
- **Model Comparison**: Performance ranking and efficiency analysis
- **Statistical Validation**: Confidence intervals and significance testing

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
```bash
# Navigate to project directory
cd "Master's Thesis"
```

2. **Create virtual environment (recommended)**
```bash
python -m venv thesis_env
# Windows
thesis_env\Scripts\activate
# Linux/Mac
source thesis_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, xgboost, catboost, torch; print('All dependencies installed successfully')"
```

## How to Run the Project

### 1. Basic Training Pipeline

**Run the complete ML pipeline:**
```bash
python train.py
```

**Run for specific city:**
```bash
# Jilin only
python train.py --which jl

# Yantai only  
python train.py --which yt
```

**Custom data paths:**
```bash
python train.py --jl_csv path/to/jilin_data.csv --yt_csv path/to/yantai_data.csv
```

### 2. Exploratory Data Analysis

**Generate comprehensive EDA plots:**
```bash
python ata_eda.py delivery_jl.csv --outdir eda_jilin_output
python ata_eda.py delivery_yt.csv --outdir eda_yantai_output
```

**Quick demo with synthetic data:**
```bash
python -c "from ata_eda import quick_demo; quick_demo()"
```

### 3. Error Analysis

**Basic error analysis:**
```bash
python error_analysis.py
```

**Detailed error analysis:**
```bash
python detailed_error_analysis.py
```

**Residual analysis:**
```bash
python residual_analysis.py
```

**Additional analysis (KNN scatter plots and Transformer validation curves):**
```bash
python additional_analysis.py
```

### 4. Complete Analysis Pipeline

**Run all analysis scripts in sequence:**
```bash
# 1. Train models
python train.py

# 2. Basic error analysis
python error_analysis.py

# 3. Detailed error analysis
python detailed_error_analysis.py

# 4. Residual analysis
python residual_analysis.py

# 5. Additional analysis
python additional_analysis.py
```

## Expected Results

### 1. Model Performance Results

**Best Performing Models:**
1. **RandomForest**: MAE = 1.276 minutes (average across cities)
2. **CatBoost**: MAE = 3.271 minutes
3. **XGBoost**: MAE = 4.304 minutes
4. **KNN**: MAE = 28.700 minutes
5. **Transformer**: MAE = 41.943 minutes
6. **Linear**: MAE = 58.687 minutes

**Key Performance Metrics:**
- **RandomForest**: R² = 0.998, Accuracy (±10min) = 98.4%
- **CatBoost**: R² = 0.990, Accuracy (±10min) = 96.6%
- **XGBoost**: R² = 0.990, Accuracy (±10min) = 93.9%

### 2. Generated Output Files

#### **Results Data:**
- `metrics.json` - Complete model performance metrics
- `metrics.csv` - Flattened results for analysis
- `metrics_pivot_Jilin.csv` - Jilin city results table
- `metrics_pivot_Yantai.csv` - Yantai city results table

#### **Analysis Plots (38+ files):**
- **Error Analysis**: Model comparison, accuracy analysis, performance gaps
- **Residual Analysis**: Q-Q plots, heteroscedasticity, outlier analysis
- **KNN Analysis**: Scatter plots, prediction accuracy
- **Transformer Analysis**: Validation curves, overfitting analysis
- **Thesis Figures**: Publication-quality plots ready for thesis


### 3. Key Findings

#### **Model Rankings:**
1. **RandomForest** - Best overall performance with excellent residuals
2. **CatBoost** - Best statistical properties (normal residuals)
3. **XGBoost** - Strong performance with good convergence
4. **KNN** - Good statistical properties but moderate accuracy
5. **Transformer** - High variance, overfitting issues
6. **Linear** - Poor performance, systematic bias

#### **Statistical Validation:**
- **Normality**: KNN, CatBoost, and XGBoost show normal residuals
- **Homoscedasticity**: All models maintain constant variance
- **Outlier Rates**: KNN and XGBoost have lowest outlier rates (0.9%)
- **Overfitting**: Transformer shows significant overfitting

#### **City Performance:**
- **Yantai** generally outperforms **Jilin** across all models
- **Larger dataset** in Yantai (182,898 vs 27,666 records) improves performance
- **Consistent patterns** across cities for model rankings

### 4. Production Recommendations

#### **Primary Model**: RandomForest
- Best overall performance (MAE: 1.276 minutes)
- Robust to outliers and noise
- Good interpretability

#### **Backup Model**: CatBoost
- Best statistical properties
- Excellent residual behavior
- Built-in categorical feature handling

#### **Monitoring Strategy**:
- Track residual distribution changes
- Monitor for increasing heteroscedasticity
- Alert on significant changes in outlier rates

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
```bash
pip install --upgrade -r requirements.txt
```

2. **Memory Issues**
- Reduce dataset size for testing
- Use smaller batch sizes for neural networks

3. **Weather API Issues**
- Check internet connection
- API requests are cached locally in `.weather_cache/`

4. **File Path Issues**
- Ensure CSV files are in the correct directory
- Use absolute paths if needed

### Performance Optimization

- **Parallel Processing**: Models use available CPU cores
- **Caching**: Weather data is cached to avoid repeated API calls
- **Memory Management**: Large datasets are processed in chunks

## Project Statistics

- **Total Code**: ~2,500 lines of Python
- **Generated Plots**: 38+ publication-quality figures
- **Statistical Tests**: 15+ different diagnostic tests
- **Dataset Size**: 210,564+ delivery records
- **Models Evaluated**: 6 different algorithms
- **Cities Analyzed**: 2 (Jilin and Yantai)

## Citation

If you use this project in your research, please cite:

```
Delivery Time Prediction Models for Last-Mile Delivery Systems
Master's Thesis - [Your Name]
[Year] - [University]
```

## License

This project is created for academic research purposes. Please ensure proper attribution when using the code or results.

## Contact

For questions about this project, please refer to the thesis documentation or contact the author.

---

**Last Updated**: [Current Date]
**Version**: 1.0
**Status**: Complete - Ready for Thesis Submission
