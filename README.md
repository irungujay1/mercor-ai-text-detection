# mercor-ai-text-detection
This notebook classifies text as human or AI-generated/copied. It performs EDA, preprocessing, and feature engineering. It uses TF-IDF Logistic Regression and LightGBM models, stacking predictions for improved results. Cross-validation and adversarial validation are included, and a submission file is created.

## Workflow Overview

The notebook follows a standard machine learning pipeline:

1.  **Setup and Libraries:** Imports all necessary libraries.
2.  **Data Loading:** Loads the training and testing datasets (`train.csv`, `test.csv`).
3.  **Exploratory Data Analysis (EDA):** Analyzes the data through basic text statistics, class balance, and visualizations like word clouds, readability scores (Flesch Reading Ease, Kincaid Grade Level), and personalization features (pronoun, emotional, and concrete word usage).
4.  **Preprocessing:** Cleans the text data by normalizing whitespace and dropping duplicate entries in the training set.
5.  **Feature Engineering:** Extracts stylometry and document-level features such as character count, word count, sentence count, average word length, type-token ratio, uppercase word fraction, and comma fraction.
6.  **TF-IDF Baseline:** Implements a baseline model using TF-IDF vectorization and Logistic Regression with Stratified K-Fold cross-validation.
7.  **LightGBM on Engineered Features:** Trains a LightGBM model on the engineered features, also using Stratified K-Fold cross-validation.
8.  **Model Evaluation (ROC-AUC):** Evaluates the performance of the individual LightGBM model and the overall stacked model using the ROC-AUC metric.
9.  **Cross Validation:** Demonstrates the performance of the LightGBM model across multiple cross-validation folds.
10. **Stacking:** Combines the out-of-fold predictions from the TF-IDF Logistic Regression and LightGBM models using a simple Logistic Regression meta-model.
11. **Calibrate probabilities:** Includes checks for per-topic AUC and adversarial validation to assess model robustness and potential data shifts.
12. **Submission:** Generates the final `submission.csv` file in the required format with predicted probabilities.

## Models Used

*   **TF-IDF with Logistic Regression:** A baseline model using term frequency-inverse document frequency features.
*   **LightGBM:** A gradient boosting model trained on engineered statistical and linguistic features.
*   **Stacked Ensemble:** A meta-model (Logistic Regression) trained on the out-of-fold predictions of the TF-IDF and LightGBM models.

## Requirements

The notebook uses standard Python libraries. The key libraries include:

*   `pandas`
*   `numpy`
*   `sklearn`
*   `lightgbm`
*   `nltk`
*   `wordcloud`
*   `matplotlib`
*   `seaborn`
*   `textstat`

Make sure these are installed in your environment. The notebook includes cells to install any necessary packages if you are running it in a fresh environment.

## Usage

1.  Ensure you have the required datasets (`train.csv`, `test.csv`, `sample_submission.csv`) in the correct directory (e.g., `/content/`).
2.  Run the notebook cells sequentially.
3.  The final `submission.csv` file will be generated in the working directory.
