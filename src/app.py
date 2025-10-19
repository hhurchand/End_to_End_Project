import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Email Spam Classifier (Pick a Model) by Sultan")
# 1) Load data
df = pd.read_csv("data/processed/CLEAN_EMAILS.csv")
X = df["email"].astype(str).values
y = df["label"].astype(int).values

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Vectorize (same for all models)
vectorizer = CountVectorizer()
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

# 4) Pick a model
model_name = st.radio("Choose model:", ["LinearSVC", "MultinomialNB", "LogisticRegression"])

if model_name == "LinearSVC":
    clf = LinearSVC()
elif model_name == "MultinomialNB":
    clf = MultinomialNB()
else:
    clf = LogisticRegression(max_iter=1000)

# 5) Train + evaluate
clf.fit(Xtr, y_train)
preds = clf.predict(Xte)
acc = accuracy_score(y_test, preds)
st.write(f"Test accuracy: **{acc:.4f}**")

st.subheader("Classification Report")
st.text(classification_report(y_test, preds, target_names=["Ham", "Spam"], digits=4))

# 6) Ham vs Spam (train)
st.subheader("Ham vs Spam (Train)")
ham = int((y_train == 0).sum())
spam = int((y_train == 1).sum())
fig1 = plt.figure()
plt.bar(["Ham", "Spam"], [ham, spam])
plt.title("Ham vs Spam (Train)")
plt.xlabel("Class"); plt.ylabel("Count")
plt.tight_layout()
st.pyplot(fig1)

# 7) Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, preds)
fig2 = plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Ham","Spam"])
plt.yticks([0,1], ["Ham","Spam"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, int(cm[i, j]), ha="center", va="center")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
st.pyplot(fig2)

# 8) User input to see ham or spam
st.subheader("Try an email yourself!")
txt = st.text_area("Paste email text here:", height=150)
if st.button("Predict"):
    vec = vectorizer.transform([txt])
    out = clf.predict(vec)[0]
    st.write("Prediction:", "**SPAM**" if out == 1 else "**HAM**")
