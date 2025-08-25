# Comprehensive ML Pipeline on Heart Disease (UCI) Dataset

This repository implements an end-to-end **Machine Learning pipeline** on the UCI Heart Disease dataset.  
The workflow covers all key steps from raw data preprocessing to model deployment with a Streamlit UI.

## Pipeline Overview
- **Data Preprocessing & Cleaning**  
  Handling missing values, encoding categorical features, scaling numerical features, and EDA.
- **Dimensionality Reduction (PCA)**  
  Reducing feature dimensionality while retaining variance.
- **Feature Selection**  
  Using Feature Importance (Random Forest), Recursive Feature Elimination (RFE), and Chi-Square tests.
- **Supervised Learning**  
  Training Logistic Regression, Decision Tree, Random Forest, and SVM classifiers.
- **Unsupervised Learning**  
  Applying K-Means and Hierarchical Clustering for pattern discovery.
- **Hyperparameter Tuning**  
  GridSearchCV and RandomizedSearchCV for optimized model performance.
- **Model Export**  
  Saving the final model pipeline (`preprocessing + best model`) as `.pkl`.
- **Interactive Streamlit UI**  
  A user-friendly app that allows input of patient health data with descriptive labels and displays predictions along with probability bars.

---

## Quick Start

1. **Dataset**  
   Place the UCI Heart Disease dataset at:
   ```
   data/heart_disease.xls
   ```
   (The Cleveland subset is commonly used).

2. **Install requirements**  
   Create and activate a virtual environment, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks**  
   Open and execute the notebooks in sequence:
   - `01_data_preprocessing.ipynb`
   - `02_pca_analysis.ipynb`
   - `03_feature_selection.ipynb`
   - `04_supervised_learning.ipynb`
   - `05_unsupervised_learning.ipynb`
   - `06_hyperparameter_tuning.ipynb`

   The final tuned model pipeline will be saved as:
   ```
   models/final_model_(full_pipeline).pkl
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run ui/app.py
   ```

   - The UI reads the schema from `ui/feature_schema.json` and displays **human-readable labels** for categorical features (e.g., Chest Pain Type, Slope of ST Segment, Thallium Test results).  
   - Predictions are displayed with a clear message:
     - ✅ High Risk of Heart Disease  
     - 🫀 Low Risk of Heart Disease  
   - The probability of heart disease is also shown as a progress bar with a percentage.

---

## Project Structure
```
Heart_Disease_Project/
├─ data/
│  └─ heart_disease.xls
├─ notebooks/
│  ├─ 01_data_preprocessing.ipynb
│  ├─ 02_pca_analysis.ipynb
│  ├─ 03_feature_selection.ipynb
│  ├─ 04_supervised_learning.ipynb
│  ├─ 05_unsupervised_learning.ipynb
│  └─ 06_hyperparameter_tuning.ipynb
├─ models/
│  └─ final_model_(full_pipeline).pkl
│  └─ best_model.pkl
│  └─ preprocessor.pkl
├─ ui/
│  ├─ app.py
│  └─ feature_schema.json
├─ results/
│  └─ evaluation_metrics.txt
│  └─ ROC Curves.png
├─ requirements.txt
└─ README.md
```

---

## Notes
- The dataset itself is not included due to licensing; users must download it from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
- The saved pipeline (`final_model_(full_pipeline).pkl`) already includes preprocessing, so the Streamlit app can directly handle raw user input.  
