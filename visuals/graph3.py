import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/spam_Emails_data.csv")
counts = df['label'].value_counts()

plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['red', 'blue'], startangle=90)
plt.title('Proportion of Emails by Label')
plt.show()
