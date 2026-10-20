# Fraudulent Job Posting Detection

**Machine Learning Based Detection of Fraudulent Job Postings**

An end-to-end machine learning project that analyzes job posting information and predicts whether a listing may be fraudulent. The project combines exploratory data analysis, natural language processing, machine learning model evaluation, and an interactive web application.

<p align="center">

[**Live Demo**](https://fake-job-posting-analyzer-agopejbjku8pyvbrwpfv8g.streamlit.app/) · [**Source Code**](https://github.com/auyashas/fake-job-posting-analyzer)

</p>

---

## Live Preview

**Try the application:** [Fraudulent Job Posting Detector](https://fake-job-posting-analyzer-agopejbjku8pyvbrwpfv8g.streamlit.app/)

The web application allows users to enter job posting details and receive a model prediction, along with an estimated fraud probability.

**Application features**
- Interactive form for job posting details
- Input validation for required fields
- Legitimate or potentially fraudulent classification
- Estimated fraud probability with a visual indicator
- Deployed web interface powered by Streamlit

> **Note:** Predictions are estimates produced by a machine learning model and should not be treated as definitive proof that a job posting is legitimate or fraudulent.

## Project Overview

Online recruitment platforms can be targeted by fraudulent job postings designed to deceive job seekers, collect personal information, or request illegitimate payments. Manually identifying suspicious listings at scale can be challenging.

This project explores how machine learning and text analysis can help identify potentially fraudulent job postings by learning patterns from job descriptions, company profiles, requirements, and structured listing attributes.

The trained model is integrated into a web application so users can submit job details and receive an immediate prediction.

## Objectives

- Analyze job posting data to identify patterns associated with fraudulent listings.
- Perform exploratory data analysis (EDA) to understand class distribution, missing values, and feature characteristics.
- Preprocess textual and categorical data for machine learning.
- Train and compare multiple classification algorithms.
- Evaluate models using metrics suited to an imbalanced classification problem.
- Deploy the selected model through an interactive web application.

## Dataset

The project uses the **Real or Fake Job Posting Prediction** dataset, commonly known as the Employment Scam Aegean Dataset (EMSCAD).

| Attribute | Details |
|---|---|
| Dataset size | 17,880 job postings |
| Number of features | 17 input features and 1 target column |
| Target column | `fraudulent` |
| Legitimate postings | 17,014 (approximately 95.16%) |
| Fraudulent postings | 866 (approximately 4.84%) |

The target variable is encoded as:
- `0` — Legitimate job posting
- `1` — Fraudulent job posting

The dataset is imbalanced, with substantially fewer fraudulent postings than legitimate ones. Consequently, model evaluation considers fraud precision, fraud recall, F1-score, and ROC-AUC in addition to accuracy.

## Methodology

The project follows an end-to-end machine learning workflow:

1. **Data loading and inspection** — Examine dataset dimensions, column types, missing values, duplicates, and target distribution.
2. **Exploratory data analysis** — Study class imbalance, categorical features, binary attributes, and text-length patterns.
3. **Data preparation** — Separate the target from the input features and create stratified training and test sets.
4. **Text preprocessing** — Combine relevant textual fields and transform them into numerical features using TF-IDF with unigram and bigram features.
5. **Categorical preprocessing** — Handle missing values and encode categorical variables using one-hot encoding.
6. **Model training** — Train and compare six classification algorithms.
7. **Evaluation and selection** — Compare model metrics and inspect confusion matrices to understand false positives and missed fraudulent postings.
8. **Model persistence** — Save the selected preprocessing-and-classification pipeline.
9. **Deployment** — Integrate the saved model into a Streamlit web application.

## Machine Learning Models

The following six models were trained and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- AdaBoost

Random Forest and AdaBoost provide ensemble-learning comparisons alongside the individual classification algorithms.

### Evaluation Results

The following results were obtained on the held-out test set:

| Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 97.90% | 72.90% | 90.17% | 80.62% | 98.96% |
| K-Nearest Neighbors | 98.10% | 90.70% | 67.63% | 77.48% | 94.01% |
| Decision Tree | 97.46% | 72.28% | 76.88% | 74.51% | 87.69% |
| Random Forest | 97.93% | 98.06% | 58.38% | 73.19% | 99.04% |
| Multinomial Naive Bayes | 96.95% | 93.24% | 39.88% | 55.87% | 95.25% |
| AdaBoost | 96.34% | 92.00% | 26.59% | 41.26% | 97.03% |

### Selected Model: Logistic Regression

Logistic Regression was selected as the primary model based on its fraud recall and fraud F1-score.

- **Fraud recall:** 90.17%
- **Fraud F1-score:** 80.62%
- **Fraud precision:** 72.90%
- **ROC-AUC:** 98.96%

The model identified 156 of the 173 fraudulent postings in the test set, while incorrectly flagging 58 legitimate postings.

Although other models achieved higher accuracy, precision, or ROC-AUC, Logistic Regression provided the strongest balance of fraud recall and F1-score among the evaluated models. Fraud recall is particularly relevant because missed fraudulent postings may expose job seekers to risk.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Matplotlib / Seaborn | Exploratory data visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |
| TF-IDF | Text feature extraction |
| Joblib | Saving and loading the trained pipeline |
| Google Colab | Model development and experimentation |
| GitHub | Version control and source code hosting |
| Streamlit | Interactive web application |
| Streamlit Community Cloud | Application deployment |

## Repository Structure

```text
fake-job-posting-analyzer/
├── app.py
├── fakeJobPostAnalyzer.ipynb
├── fraudulent_job_detector.joblib
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit application and prediction interface.
- `fakeJobPostAnalyzer.ipynb` — Data analysis, preprocessing, model training, evaluation, and model saving.
- `fraudulent_job_detector.joblib` — Serialized selected model pipeline.
- `requirements.txt` — Python package dependencies.
- `README.md` — Project overview and usage documentation.

The dataset is obtained separately through the configured dataset download workflow and may not be stored directly in this repository.

## Run Locally

### Prerequisites

- Python installed on your system
- Git
- Access to the trained model file

### 1. Clone the repository

```bash
git clone https://github.com/auyashas/fake-job-posting-analyzer.git
cd fake-job-posting-analyzer
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For compatibility with the serialized model, use a Python environment and scikit-learn version compatible with the environment used to train and save it.

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where the application can be accessed in your browser.

## Limitations and Responsible Use

- Model predictions are probabilistic estimates, not definitive fraud determinations.
- The dataset is imbalanced, and performance on unseen or changing real-world job postings may differ from test-set results.
- Fraud probability is not necessarily a calibrated estimate of real-world fraud prevalence.
- Text, missing values, categorical features, and differences between training data and new postings can affect predictions.
- Users should independently verify employers, contact details, websites, and requests for payment or sensitive information.

This project is intended for educational and research purposes and should support, rather than replace, careful human judgment.

## Future Improvements

- Evaluate the model on newer and independently collected job postings.
- Tune the classification threshold using validation data.
- Investigate probability calibration and explainability techniques.
- Improve handling of previously unseen categorical values and incomplete postings.
- Add automated testing, application monitoring, and more comprehensive input validation.
- Explore additional models and approaches for text classification.

## Contributors

This project was developed collaboratively.

| Contributor | GitHub |
|---|---|
| Contributor 1 | [@auyashas](https://github.com/auyashas) |
| Contributor 2 | [@thanisha2060](https://github.com/thanisha2060) |


## Acknowledgements

Thanks to the dataset creators and the open-source Python machine learning community for the data and tools used in this project.

---

<p align="center">
  <strong>Machine Learning Based Detection of Fraudulent Job Postings</strong><br>
  Built with Python, Scikit-learn, and Streamlit.
</p>
