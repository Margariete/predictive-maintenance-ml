# Predictive Maintenance Using Machine Learning

## Skills Demonstrated

- Python programming (pandas, scikit-learn)
- Data cleaning and preprocessing
- Exploratory data analysis and visualization
- Machine learning model development (Random Forest)
- Model evaluation for imbalanced classification problems
- Identification and correction of data leakage
- Precision–recall tradeoff analysis and threshold tuning
- Engineering-focused interpretation of model performance

---

## Overview

This project develops a machine learning model to predict equipment failure using industrial sensor data. The goal is to build an interpretable and realistic predictive maintenance workflow, emphasizing not only model performance but also sound engineering evaluation.

The analysis highlights key challenges in real-world ML, including class imbalance, data leakage, and the tradeoff between precision and recall.

---

## Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset, which includes:

- Air temperature  
- Process temperature  
- Rotational speed  
- Torque  
- Tool wear  
- Machine failure labels  

---

## Key Methods

- Exploratory data analysis (EDA)
- Feature relationship visualization
- Random Forest classification
- Data leakage detection and correction
- Confusion matrix and classification report evaluation
- Threshold tuning to improve failure detection

---

## Key Results

- Initial model achieved near-perfect performance due to data leakage  
- After removing leakage, model performance became realistic  
- Default model recall for failures: **34%**
- Tuned model recall (threshold = 0.30): **70%**
- Improved detection of failures at the cost of increased false positives  

This tradeoff is appropriate for predictive maintenance, where missing failures is more costly than raising false alarms.

---

## Project Structure


- predictive-maintenance-ml/
 - data/
 - figures/
 - src/
  -  01_data_exploration.py
  -  02_modeling_leakage_and_threshold.py
  - README.md
  - requirements.txt
  - .gitignore


---

## How to Run

Install dependencies:


- bash
 - pip install -r requirements.txt

- Run data exploration:
 - python3 src/01_data_exploration.py

- Run modeling workflow:
 - python3 src/02_modeling_leakage_and_threshold.py


## Future Improvements

- Add feature importance visualization
- Plot precision-recall curves
- Compare Random Forest with logistic regression
- Add cross-validation
- Package reusable functions into a cleaner Python module

