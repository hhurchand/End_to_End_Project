import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

class Visualization:
    def __init__(self, df, label_col="label", text_col="Message",output_dir="reports/figures"):
        """
        Loads data and sets label and text columns.
        """
        self.df = df
        self.label_col = label_col
        self.text_col = text_col
        self.output_dir = output_dir

    def plot_distribution(self):
        """
        Plots a pie chart of the label distribution (spam vs. ham).
        """
        category_counts = self.df[self.label_col].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(
            category_counts,
            labels=category_counts.index.astype(str),
            autopct='%1.1f%%',
            startangle=140
        )
        plt.title("Distribution of Spam vs. Ham")
        plt.axis("equal")
        output_path = os.path.join(self.output_dir, "pie_chart.jpeg")
        plt.savefig(output_path)
        plt.close()

    def plot_wordclouds(self):
        """
        Generates and displays word clouds for each category.
        """
        categories = self.df[self.label_col].unique()
        for category in categories:
            filtered_df = self.df[self.df[self.label_col] == category]
            text = ' '.join(filtered_df[self.text_col].dropna().astype(str))
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.title(f"Word Cloud for Category: {category}")
            plt.axis("off")
            category_str = str(category).strip().replace(' ', '_').replace('"', '').replace("'", '')
            filename = f"wordcloud_label_{category_str}.jpeg"
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path)
            plt.close()

    def visualize(self):
        """
        Run both pie chart and word cloud visualizations.
        """
        self.plot_distribution()
        self.plot_wordclouds()
