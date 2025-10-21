Email Spam or Ham Checker
=========================

A complete end-to-end machine learning pipeline by **H.Patel** that detects whether an email is **Spam** or **Ham** using TF-IDF features and three classic ML algorithms.  
This documentation was generated automatically with Sphinx and includes data processing, model training, evaluation results, and usage instructions.

---

📘 Project Overview
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
6. Streamlit web app for predictions  

---

📊 Data Description
-------------------

- Dataset 1 → ``data/combined_dataset.csv``  
- Dataset 2 → ``data/enron_spam_data.csv``  
- Merged output → ``data/merged_spam.csv``  

After merging, the dataset contains **44,003 samples** and **7 columns**.  
Preprocessing removes duplicates, converts all text to lowercase, and drops null values.

---

⚙️ Training Workflow
---------------------

1. **Merge Datasets**  
   Combines ``combined_dataset.csv`` and ``enron_spam_data.csv`` → ``merged_spam.csv``  

2. **Configuration**  
   Parameters are loaded from ``params.yaml`` and updated dynamically before training.  

3. **TF-IDF Vectorization**  
   - `min_df = 2`  
   - `max_df = 0.95`  
   - `ngram_range = (1, 2)`  

4. **Model Training**  
   Trains and evaluates:  
   - Logistic Regression  
   - Linear SVM  
   - Random Forest  
   Each uses GridSearchCV (3-fold, scoring = F1) and logs metrics to MLflow.  

5. **Model Saving**  
   Saves all trained models and TF-IDF vectorizer in the ``model/`` folder.  

6. **Prediction**  
   Loads vectorizer + model, predicts **Spam** or **Ham**, and returns probability.

---

🏆 Model Performance (latest run)
---------------------------------

Results from the latest training run (`data/merged_spam.csv`, 44,003 samples):

+--------------------+-----------+-----------+
| Model              | Accuracy  | F1 Score  |
+====================+===========+===========+
| Logistic Regression| 0.9504    | 0.8901    |
+--------------------+-----------+-----------+
| Linear SVM         | 0.9665    | 0.9224    |
+--------------------+-----------+-----------+
| Random Forest      | 0.9427    | 0.8543    |
+--------------------+-----------+-----------+

✅ **Best Model:** **Linear SVM** (F1 = **0.9224**, Accuracy = **0.9665**)

---

📈 Evaluation Metrics
---------------------

Each model logs metrics via **MLflow**, including:
- Accuracy  
- Precision  
- Recall  
- F1 Score  

All models are compared and summarized in the terminal output after training.

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

🧠 Example Commands
-------------------

.. code-block:: bat

   cd C:\Users\Owner\OneDrive\Desktop\final_final_project\Project_final
   python main.py
   REM → trains all 3 models & logs metrics to MLflow

   python -m src.models.predict
   REM → loads saved model and makes prediction on new email text

---

📊 MLflow Tracking
-------------------

1. Start MLflow UI:
   .. code-block:: bat
      mlflow ui

2. Then visit:
   http://localhost:5000

View metrics, parameters, and model comparisons for each run.

---

🧩 Streamlit Application
------------------------

Launch the visual web app for testing predictions interactively:
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
