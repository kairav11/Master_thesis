# Conceptual Flow of Delivery Time Prediction Pipeline
## Master's Thesis Project - Complete Pipeline Overview

## 1. Overall Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DELIVERY TIME PREDICTION PIPELINE                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Raw Data → Data Processing → Feature Engineering → ML Training → Analysis  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Detailed Pipeline Flow

### Phase 1: Data Collection & Preprocessing
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Raw CSV   │ →  │ Data Loading │ →  │ Data        │ →  │ Data         │
│   Files     │    │ (loaders.py) │    │ Cleaning    │    │ Validation   │
│             │    │              │    │ (quality.py)│    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
     │                     │                    │                    │
     ▼                     ▼                    ▼                    ▼
delivery_jl.csv    Load both cities      Remove duplicates    Validate GPS
delivery_yt.csv    (27,666 + 182,898)    Handle missing      coordinates
                                      values, outliers
```

### Phase 2: Feature Engineering
```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Temporal   │ →  │ Distance    │ →  │ Weather      │ →  │ Derived     │
│   Features   │    │ Features    │    │ Features     │    │ Features    │
│              │    │             │    │              │    │             │
│ • Hour       │    │ • Haversine │    │ • Temperature│    │ • Speed     │
│ • Day of     │    │   distance  │    │ • Rain       │    │ • Buckets   │
│   week       │    │ • GPS       │    │ • Wind       │    │ • Flags     │
│ • Holidays   │    │   accuracy  │    │ • API data   │    │ • Encoding  │
└──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

### Phase 3: Machine Learning Pipeline
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Feature   │ →  │ Train/Test   │ →  │ Model       │ →  │ Model        │
│   Selection │    │ Split        │    │ Training    │    │ Evaluation   │
│             │    │              │    │             │    │              │
│ • Numeric   │    │ • 80/20      │    │ • 6 Models  │    │ • Metrics    │
│   features  │    │   split      │    │   trained   │    │   computed   │
│ • Categorical│    │ • Random     │    │ • Cross-    │    │ • Validation │
│   encoding  │    │   state=42   │    │   validation│    │   performed  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### Phase 4: Model Evaluation & Analysis
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Error     │ →  │ Residual     │ →  │ Statistical │ →  │ Thesis       │
│   Analysis  │    │ Analysis     │    │ Validation  │    │ Reports      │
│             │    │              │    │             │    │              │
│ • MAE/RMSE  │    │ • Q-Q plots  │    │ • Normality │    │ • 38+ plots  │
│ • R²/Acc    │    │ • Hetero-    │    │   tests     │    │ • 4 reports  │
│ • Rankings  │    │   scasticity │    │ • Confidence│    │ • LaTeX      │
│ • Gaps      │    │ • Outliers   │    │   intervals │    │   tables     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

## 3. Model Architecture Flow

### Classical Models Pipeline
```
Input Features → Preprocessing → Model Training → Prediction → Evaluation
     │               │               │              │           │
     ▼               ▼               ▼              ▼           ▼
[Distance,      [Standardize,   [Random Forest,  [Predict    [MAE, RMSE,
 Hour,          Impute,         XGBoost,         delivery    R², Accuracy]
 Temperature,   Encode]         CatBoost,        times]      metrics]
 Weather,       ]               KNN,             ]
 etc.]                          Linear]          ]
```

### Neural Network Pipeline (Transformer)
```
Input Features → Embedding → Transformer → Output Head → Prediction
     │             │            │            │             │
     ▼             ▼            ▼            ▼             ▼
[Numeric      [Linear        [Multi-head   [Linear +     [Delivery
 features]    projection]    attention,    ReLU]         time
              ]              layers]       ]             prediction]
```

## 4. Analysis Pipeline Flow

### Error Analysis Flow
```
Model Results → Basic Analysis → Detailed Analysis → Residual Analysis
      │               │                │                   │
      ▼               ▼                ▼                   ▼
[Metrics]      [Performance    [Efficiency,      [Normality,
 JSON]         comparisons]    consistency]      heteroscedasticity,
               38+ plots       5 plots           outliers]
               4 reports       1 JSON            7 plots
                                1 CSV             2 JSON/CSV
```

### Statistical Validation Flow
```
Raw Results → Statistical Tests → Validation → Confidence Intervals
     │              │                 │              │
     ▼              ▼                 ▼              ▼
[MAE, RMSE,    [Normality,      [Cross-val,     [95% CI for
 R², Acc]      heteroscedasticity, Bootstrap]   all metrics]
               outlier tests]                    ]
```

## 5. Output Generation Flow

### File Generation Hierarchy
```
Main Scripts → Analysis Scripts → Output Directories → Final Reports
     │               │                   │                   │
     ▼               ▼                   ▼                   ▼
[train.py]      [error_analysis.py,  [error_analysis/,    [README.md,
               detailed_error_       detailed_analysis/,  CONCEPTUAL_FLOW.md,
               analysis.py,          residual_analysis/,  thesis_*_report.md]
               residual_analysis.py, additional_analysis/, 
               additional_analysis.py] thesis_figures/,
                              thesis_summary/]
```

## 6. Data Flow Between Components

### Input Data Flow
```
Raw Data Files → Data Loaders → Feature Engineering → Model Training
     │              │                │                    │
     ▼              ▼                ▼                    ▼
[CSV files]    [loaders.py]     [engineering.py]    [classical.py,
27,666 JL      loads and        creates features    transformers.py]
182,898 YT     validates        and weather data    trains 6 models
records]       data]            integration]        ]
```

### Results Data Flow
```
Model Training → Results Storage → Analysis Processing → Report Generation
      │                │                  │                    │
      ▼                ▼                  ▼                    ▼
[6 models        [metrics.json,     [Statistical        [38+ plots,
 trained]        metrics.csv,       analysis,          4 reports,
                 pivot tables]      visualization]      LaTeX tables]
```

## 7. Key Processing Steps

### Step 1: Data Preprocessing
1. **Load raw CSV files** (Jilin: 27,666 records, Yantai: 182,898 records)
2. **Clean data** (remove duplicates, handle missing values)
3. **Validate GPS coordinates** (ensure within reasonable bounds)
4. **Parse timestamps** (convert to datetime format)

### Step 2: Feature Engineering
1. **Temporal features** (hour, day of week, holidays)
2. **Distance calculation** (Haversine formula for GPS coordinates)
3. **Weather integration** (Open-Meteo API for temperature, precipitation, wind)
4. **Derived features** (speed, buckets, categorical encoding)

### Step 3: Model Training
1. **Split data** (80% training, 20% testing)
2. **Train 6 models** (Linear, KNN, RandomForest, XGBoost, CatBoost, Transformer)
3. **Cross-validation** (5-fold CV for robust evaluation)
4. **Hyperparameter tuning** (where applicable)

### Step 4: Evaluation & Analysis
1. **Calculate metrics** (MAE, RMSE, R², Accuracy)
2. **Statistical analysis** (normality tests, residual analysis)
3. **Generate visualizations** (38+ publication-quality plots)
4. **Create reports** (4 comprehensive analysis reports)

## 8. Performance Optimization Flow

### Computational Efficiency
```
Data Processing → Model Training → Analysis → Visualization
      │               │              │            │
      ▼               ▼              ▼            ▼
[Chunked        [Parallel        [Vectorized    [Batch
 processing,    training,        operations,    plotting,
 caching]       GPU usage]       caching]      optimization]
```

### Memory Management
- **Chunked processing** for large datasets
- **Caching** for weather API calls
- **Efficient data structures** (pandas, numpy)
- **Memory cleanup** after each analysis

## 9. Quality Assurance Flow

### Validation Pipeline
```
Input Validation → Processing Validation → Output Validation → Report Validation
       │                    │                     │                   │
       ▼                    ▼                     ▼                   ▼
[Data quality     [Feature engineering  [Model performance  [Statistical
 checks,          validation,           validation,         validation,
 format checks]   range checks]         metric checks]      significance tests]
```

### Error Handling
- **Try-catch blocks** for API calls
- **Graceful degradation** for missing data
- **Validation checks** at each pipeline stage
- **Comprehensive logging** for debugging

## 10. Reproducibility Flow

### Reproducible Research
```
Random Seeds → Version Control → Documentation → Dependencies
     │              │                │              │
     ▼              ▼                ▼              ▼
[Fixed seeds    [Git tracking,    [README,       [requirements.txt,
 across all     code comments]    CONCEPTUAL_    environment setup]
 scripts]                          FLOW.md]      ]
```

## Summary

This conceptual flow demonstrates a comprehensive machine learning pipeline that:

1. **Processes** real-world delivery data from two cities
2. **Engineers** sophisticated features including weather data
3. **Trains** six different machine learning models
4. **Evaluates** performance using multiple statistical methods
5. **Generates** publication-quality analysis and reports
6. **Ensures** reproducibility and scientific rigor

The pipeline produces **38+ plots**, **4 comprehensive reports**, and **detailed statistical validation** suitable for a Master's thesis in machine learning and data science.

---

**Total Processing Time**: ~30-45 minutes for complete pipeline
**Output Size**: ~500MB of analysis files and plots
**Statistical Rigor**: 15+ different diagnostic tests
**Publication Quality**: 300 DPI plots, LaTeX tables, comprehensive reports
