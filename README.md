# Alzheimer’s Disease Prediction Project
Using machine learning, I analyzed a dataset with information about patients with Alzheimer's Disease.

## Overview

In this project, I explore the predictive potential of a synthetic dataset containing extensive health information from 2,149 patients. The dataset provides valuable data, including demographic details, cognitive assessments, medical history, lifestyle habits, and clinical measurements. These features make it ideal for investigating the relationships between these variables and the onset and progression of Alzheimer's Disease. Key attributes in the dataset include:

- Cognitive scores (e.g., Mini-Mental State Examination, MMSE)
- Functional assessments
- Symptoms such as confusion and forgetfulness
- Diagnosis information

The goal of this project is to apply AI and machine learning techniques to uncover patterns and potential early indicators of Alzheimer's Disease. This aligns with my long-standing interest in Alzheimer’s research, and I aim to leverage these techniques to contribute to advancements in the field.

### Dataset Source

This dataset was obtained from Kaggle, which allows for comparisons with other researchers and coders in the field. The dataset offers an opportunity to apply various machine learning models to explore Alzheimer's Disease prediction and develop insights that may aid in early diagnosis.

## Code Overview

In this project, I used Python to build and evaluate machine learning models. Visual Studio Code was my primary development environment, and the code is organized into several sections:

1. **Data Preprocessing**: 
   - Data cleaning, including handling missing values and converting categorical variables into numerical values.
   - Feature selection to identify the most relevant predictors for the model.
   
2. **Model Training and Evaluation**: 
   - The code trains several machine learning models on the dataset. I experimented with a variety of algorithms, including:
     - **Support Vector Machine (SVM)**
     - **Gradient Boosting**
     - **XGBoost**
     - **Random Forest**
     - **Logistic Regression**
     - **Naive Bayes**
     - **Neural Networks**
   - For each model, I evaluated performance using metrics such as accuracy, precision, recall, and F1-score.

3. **Model Performance Analysis**:
   - The models were assessed based on their ability to accurately classify Alzheimer’s Disease in the dataset.
   - A summary of the model performance is included, highlighting the best-performing models and their strengths.

### Key Findings

- **SVM** performed impressively, achieving strong classification results.
- **Gradient Boosting**, **XGBoost**, and **Random Forest** showed excellent performance, with XGBoost emerging as the best-performing model overall.
- **Logistic Regression** and **Naive Bayes** did not perform as well, with their results lagging behind other models.
- **Neural Networks** performed the worst, as the dataset was not suitable for such complex models.
  
Based on these results, I recommend using **XGBoost** for deployment, as it consistently outperformed the other models.

## Future Work

In the future, I plan to:

- Enhance the model with more feature engineering to improve predictions.
- Experiment with deep learning techniques for better accuracy on more complex datasets.
- Publish the code in a **Google Colab** notebook to make it easier for others to view, run, and experiment with the code.

## Requirements

To run this project, the following Python libraries are required:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `tensorflow` (for neural network models)

You can install the necessary packages using the following:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow

```
## How to Run

1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Run the Jupyter Notebook (`.ipynb`) file to execute the code and view the results.

## Conclusion

This project showcases the power of machine learning in predicting Alzheimer's Disease using real-world health data. The performance analysis of different models provides insight into which algorithms are most suitable for this type of classification task. Through this work, I hope to contribute to the ongoing research efforts aimed at early detection and better understanding of Alzheimer's Disease.

Feel free to explore the code and make improvements or modifications. If you have any questions or suggestions, don't hesitate to reach out.

## Contact

You can reach me at farihankha@gmail.com.
