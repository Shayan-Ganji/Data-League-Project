import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualization(df, target='G3'):
    print("📊 Generating visualizations...")
    sns.set(style="whitegrid")
    if target not in df.columns:
        print("⚠️ Target column not found, skipping target visualization.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    

    sns.histplot(df[target], kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title(f'Distribution of {target}')
    if 'Weighted_Score' in df.columns:
        sns.scatterplot(x='Weighted_Score', y=target, data=df, ax=axes[0, 1], color='green')
        axes[0, 1].set_title('Weighted Score vs G3')
    if 'Grade_Trend' in df.columns:
        sns.boxplot(x=df['Grade_Trend'], y=df[target], ax=axes[1, 0], palette='coolwarm', orient='h')
        axes[1, 0].set_title('Impact of Grade Trend on G3')

    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        cols = corr.nlargest(10, target)[target].index
        sns.heatmap(df[cols].corr(), annot=True, cmap='viridis', fmt='.2f', ax=axes[1, 1])
        axes[1, 1].set_title('Top 10 Correlated Features')
    plt.tight_layout()
    plt.show()
    print("✅ Visualization done.")