import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/spam_Emails_data.csv")
df['length'] = df['text'].fillna("").apply(len)

df.groupby('label')['length'].mean().plot(kind='bar', color='orange')
plt.xlabel('Label')
plt.ylabel('Average Email Length')
plt.title('Average Email Length per Label')
plt.show()
