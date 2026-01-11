import matplotlib.pyplot as plt
import seaborn as sns

def plot_attrition_distribution(df):
    sns.countplot(x="Attrition", data=df)
    plt.title("Employee Attrition Distribution")
    plt.show()


def plot_income_vs_attrition(df):
    sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
    plt.title("Monthly Income by Attrition")
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
