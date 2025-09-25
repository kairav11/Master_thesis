# Quick Execution Guide
## How to Run the Complete Delivery Time Prediction Pipeline

## 🚀 Quick Start (5 minutes)

### 1. Setup Environment
```powershell
# Navigate to project directory
cd "Master's Thesis"

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```powershell
# Run all analysis scripts in sequence
python train.py; python error_analysis.py; python detailed_error_analysis.py; python residual_analysis.py; python additional_analysis.py
```

**Expected Runtime**: 30-45 minutes
**Output**: 38+ plots, 4 reports, complete analysis

## 📊 Individual Script Execution

### Core Training Pipeline
```powershell
# Basic model training and evaluation
python train.py

# Output: metrics.json, metrics.csv, pivot tables
# Runtime: ~15 minutes
```

### Analysis Scripts (Run after train.py)

#### 1. Basic Error Analysis
```powershell
python error_analysis.py
# Output: error_analysis/ directory with 6 plots
# Runtime: ~5 minutes
```

#### 2. Detailed Error Analysis
```powershell
python detailed_error_analysis.py
# Output: detailed_analysis/ directory with 5 plots
# Runtime: ~5 minutes
```

#### 3. Residual Analysis
```powershell
python residual_analysis.py
# Output: residual_analysis/ directory with 7 plots
# Runtime: ~8 minutes
```

#### 4. Additional Analysis (KNN & Transformer)
```powershell
python additional_analysis.py
# Output: additional_analysis/ directory with 8 plots
# Runtime: ~7 minutes
```

## 🎯 Expected Results Summary

### Model Performance Rankings
1. **RandomForest**: MAE = 1.276 min, R² = 0.998, Accuracy = 98.4%
2. **CatBoost**: MAE = 3.271 min, R² = 0.990, Accuracy = 96.6%
3. **XGBoost**: MAE = 4.304 min, R² = 0.990, Accuracy = 93.9%
4. **KNN**: MAE = 28.700 min, R² = 0.853, Accuracy = 35.0%
5. **Transformer**: MAE = 41.943 min, R² = 0.803, Accuracy = 21.4%
6. **Linear**: MAE = 58.687 min, R² = 0.733, Accuracy = 15.8%

### Generated Output Files

#### 📈 Analysis Plots (38+ files)
- **Error Analysis**: 6 plots (model comparison, accuracy analysis)
- **Detailed Analysis**: 5 plots (error patterns, efficiency analysis)
- **Residual Analysis**: 7 plots (Q-Q plots, heteroscedasticity)
- **Additional Analysis**: 8 plots (KNN scatter, Transformer curves)
- **Thesis Figures**: 7 plots (publication-quality)
- **EDA Plots**: 14+ plots (exploratory data analysis)

#### 📊 Data Files
- `metrics.json` - Complete model performance metrics
- `metrics.csv` - Flattened results for analysis
- `metrics_pivot_Jilin.csv` - Jilin results table
- `metrics_pivot_Yantai.csv` - Yantai results table


## 🔍 Key Findings

### Best Models
- **RandomForest**: Best overall performance, robust to outliers
- **CatBoost**: Best statistical properties (normal residuals)
- **XGBoost**: Strong performance with good convergence

### Statistical Validation
- **Normality**: KNN, CatBoost, XGBoost show normal residuals
- **Homoscedasticity**: All models maintain constant variance
- **Outlier Rates**: KNN and XGBoost have lowest outlier rates (0.9%)
- **Overfitting**: Transformer shows significant overfitting

### City Performance
- **Yantai** generally outperforms **Jilin** across all models
- **Larger dataset** in Yantai (182,898 vs 27,666 records) improves performance
- **Consistent patterns** across cities for model rankings

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### 1. Missing Dependencies
```powershell
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Check specific packages
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, xgboost, catboost, torch; print('All OK')"
```

#### 2. Memory Issues
```powershell
# For large datasets, reduce batch size in neural networks
# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')"
```

#### 3. Weather API Issues
```powershell
# Check internet connection
# Weather data is cached in .weather_cache/ directory
# Delete cache to force refresh: Remove-Item -Recurse -Force .weather_cache/
```

#### 4. File Path Issues
```powershell
# Ensure CSV files are in correct directory
Get-ChildItem delivery_jl.csv, delivery_yt.csv

# Use absolute paths if needed
python train.py --jl_csv "C:\full\path\to\delivery_jl.csv"
```

## 📋 Verification Checklist

After running the complete pipeline, verify you have:

- [ ] `metrics.json` and `metrics.csv` files
- [ ] `metrics_pivot_Jilin.csv` and `metrics_pivot_Yantai.csv`
- [ ] `error_analysis/` directory with 6 plots
- [ ] `detailed_analysis/` directory with 5 plots
- [ ] `residual_analysis/` directory with 7 plots
- [ ] `additional_analysis/` directory with 8 plots
- [ ] `thesis_figures/` directory with 7 plots
- [ ] EDA directories with exploratory plots

## 🎓 For Thesis Integration

### Essential Files for Thesis
1. **Results Section**: Use `thesis_figures/` plots
2. **Error Analysis**: Use `error_analysis/` and `residual_analysis/` plots
3. **Model Comparison**: Use `additional_analysis/` comparison plots
4. **Statistical Validation**: Reference JSON files for exact metrics

### Publication-Ready Plots
- All plots are 300 DPI, publication quality
- Consistent styling and formatting
- Professional color schemes and legends
- Ready for direct thesis integration

## 📊 Performance Benchmarks

### Expected Runtime
- **Complete Pipeline**: 30-45 minutes
- **Individual Scripts**: 5-15 minutes each
- **Memory Usage**: 2-4 GB peak
- **Output Size**: ~500 MB total

### System Requirements
- **RAM**: 4+ GB recommended
- **Storage**: 1+ GB free space
- **CPU**: Multi-core recommended
- **Python**: 3.8+ required

---

**Status**: ✅ Complete and Ready for Thesis Submission
**Quality**: 🏆 Publication-ready analysis and plots
**Coverage**: 📊 Comprehensive statistical validation
