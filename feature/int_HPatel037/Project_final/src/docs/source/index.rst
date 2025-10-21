Email Spam or Ham Checker
=========================

A complete end-to-end machine learning pipeline by **H.Patel** that detects whether an email is **Spam** or **Ham** using TF-IDF features and three classic ML algorithms.  
This documentation was generated automatically with Sphinx and includes data processing, model training, and evaluation details.

---

📂 Project Overview
-------------------

**Project Name:** Email Spam or Ham Checker  
**Author:** H.Patel  
**Version:** 1.0  
**Environment:** PFvenv (Python 3.10+)  

This project performs:
1. Data ingestion and cleaning  
2. TF-IDF feature extraction  
3. Model training (Logistic Regression, Linear SVM, Random Forest)  
4. MLflow experiment tracking  
5. Model evaluation and storage  
6. Interactive Streamlit App for predictions  

---

📊 Data Description
-------------------

- Dataset 1 → ``data/combined_dataset.csv``  
- Dataset 2 → ``data/enron_spam_data.csv``  
- Merged output → ``data/merged_spam.csv``  

The merged dataset contains thousands of email texts labeled as *Spam* (1) or *Ham* (0).  
All preprocessing removes duplicates, converts text to lowercase, and drops missing values.

---

⚙️ Training Workflow
---------------------

1. **Merge Datasets**  
   Combines ``combined_dataset.csv`` and ``enron_spam_data.csv`` → ``merged_spam.csv``.  

2. **Configuration**  
   Loaded from ``params.yaml`` and updated dynamically during runtime.  

3. **TF-IDF Vectorization**  
   Uses unigrams and bigrams with parameters:  
   - `min_df=2`  
   - `max_df=0.95`  
   - `ngram_range=(1, 2)`  

4. **Model Training**  
   Trains and evaluates:  
   - Logistic Regression  
   - Linear SVM  
   - Random Forest  
   Each model runs ``GridSearchCV`` and logs metrics to MLflow.  

5. **Model Saving**  
   All trained models and vectorizers are saved in the ``model/`` directory.  

6. **Prediction**  
   The predictor loads ``vectorizer.joblib`` and ``*_model.joblib`` to classify new emails as **Spam** or **Ham**.

---

📈 Evaluation Metrics
---------------------

Each model logs metrics via MLflow, including:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

A comparison report is printed after training to identify the **best model by F1 score**.

---

🧠 Example: Training Commands
-----------------------------

.. code-block:: bat

   cd C:\Users\Owner\OneDrive\Desktop\final_final_project\Project_final
   python main.py
   REM trains models, merges datasets, and logs results to MLflow

   python -m src.models.predict
   REM test a single email input using a saved model

---

💾 File Structure Summary
-------------------------

.. code-block:: text

   FINAL_FINAL_PROJECT/
   ├── PFvenv/
   ├── Project_final/
   │   ├── app/
   │   │   └── streamlit_app.py
   │   ├── data/
   │   │   ├── combined_dataset.csv
   │   │   ├── enron_spam_data.csv
   │   │   └── merged_spam.csv
   │   ├── model/
   │   │   ├── rf_model.joblib
   │   │   ├── logreg_model.joblib
   │   │   ├── linearsvc_model.joblib
   │   │   └── vectorizer.joblib
   │   ├── src/
   │   │   ├── data/make_dataset.py
   │   │   ├── models/tri_model_trainer.py
   │   │   ├── models/predict.py
   │   │   └── main.py
   │   ├── params.yaml
   │   └── requirements.txt

---

💡 How to View Results
-----------------------

1. After running ``python main.py``, open your MLflow UI:
   .. code-block:: bat
      mlflow ui

2. Then visit in your browser:
   http://localhost:5000

   You’ll see model runs, parameters, and metrics.

---

🧩 Interactive App
------------------

To run the **Streamlit App** (optional GUI):
.. code-block:: bat

   cd Project_final/app
   streamlit run streamlit_app.py

---

📚 API Reference
----------------
.. toctree::
   :maxdepth: 2
   :caption: Source Code Modules

   api/modules
