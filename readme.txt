# CardioCare-AI: Heart Health Prediction

An AI-driven diagnostic tool that leverages multiple machine learning models to predict cardiovascular health risks based on patient data.

## 📌 Project Overview
CardioCare-AI compares the performance of various classification algorithms to find the most accurate method for heart disease prediction. The project includes a web application interface (`app.py`) to make the models accessible for real-time testing.

## 📊 Models Included
The project evaluates and provides pre-trained versions of the following algorithms:
* **Random Forest:** (`random_forest_model.pkl`) - Used for high-accuracy ensemble learning.
* **Logistic Regression:** (`logistic_regression_model.pkl`) - Used as a baseline for binary classification.
* **Decision Tree:** (`decision_tree_model.pkl`) - Used for interpretable, rule-based logic.

## 📂 File Structure
* `Task1 (2).ipynb`: The primary research notebook containing data preprocessing, model training, and evaluation.
* `app.py`: A Python-based web interface (likely Flask or Streamlit) to interact with the models.
* `*.pkl`: Serialized model files for instant deployment without retraining.
* `tutorial vid.mkv`: A video demonstration of how the system operates.
* `readme.txt`: Original project notes.

## 🛠️ Requirements
To run the application, you will need:
* `Python 3.x`
* `scikit-learn` (to load the .pkl files)
* `pandas` & `numpy`
* `Flask` or `Streamlit` (depending on the `app.py` framework)

## 🚀 Getting Started
1. **Model Exploration:** Open `Task1 (2).ipynb` to see the data analysis and training process.
2. **Run the App:** Execute `python app.py` in your terminal to launch the user interface.
3. **Watch the Demo:** Refer to `tutorial vid.mkv` for a visual walkthrough of the features.
