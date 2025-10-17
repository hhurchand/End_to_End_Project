import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/spam_Emails_data.csv")

# Count emails per label
counts = df['label'].value_counts()

# Plot
counts.plot(kind='bar', color='skyblue')
plt.xlabel('Label')
plt.ylabel('Number of Emails')
plt.title('Emails per Label')
plt.show()
