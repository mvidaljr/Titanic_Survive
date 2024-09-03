# Titanic Survival Prediction

## Project Overview

This project focuses on predicting the survival of passengers aboard the Titanic using machine learning models. The goal is to develop a model that can accurately classify whether a passenger would survive based on various features such as age, gender, class, and more. The project achieves a high level of accuracy using the GradientBoostingClassifier.

## Dataset

- **Source:** The dataset is downloaded from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).
- **Data:** The dataset includes information about passengers such as their age, gender, ticket class, fare, and whether they survived.

## Tools & Libraries Used

- **Data Handling:**
  - `Pandas` for data manipulation and preprocessing.
  - `NumPy` for numerical operations.
- **Data Visualization:**
  - `Matplotlib` and `Seaborn` for creating plots and visualizations.
- **Machine Learning:**
  - `scikit-learn` for building and evaluating the model, specifically using the GradientBoostingClassifier.

## Methodology

### Data Preprocessing:

- **Handling Missing Values:**
  - Imputed missing values for features like age and embarked status.
  
- **Feature Engineering:**
  - Created new features and transformed existing ones to improve model performance, such as encoding categorical variables.

### Model Development:

- **Gradient Boosting Classifier:**
  - Implemented the GradientBoostingClassifier for its ability to boost the performance of weak learners, leading to a highly accurate predictive model.
  
- **Model Training:**
  - Trained the model on the preprocessed dataset, optimizing it with hyperparameters to achieve the best performance.

### Model Evaluation:

- **Accuracy:**
  - The model achieved an impressive accuracy of 0.9928, demonstrating its effectiveness in predicting survival on the Titanic.

- **Example Usage:**
  ```python
  from sklearn.ensemble import GradientBoostingClassifier
  
  model = GradientBoostingClassifier()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  ```

## Results

The GradientBoostingClassifier achieved an accuracy of 0.9928, indicating a highly effective model for predicting Titanic passenger survival based on the available features.

## Conclusion

This project successfully demonstrates the use of GradientBoostingClassifier to predict passenger survival on the Titanic with a high level of accuracy. The model's performance highlights the importance of data preprocessing and feature engineering in building effective predictive models.

## Future Work

- Experiment with other machine learning models like RandomForestClassifier or XGBoost to compare performance.
- Conduct further feature engineering to uncover hidden patterns in the data.
- Deploy the model in a web application for real-time predictions.
