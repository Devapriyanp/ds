import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(".csv")


print("First 5 rows:\n", df.head())

print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Mode:\n", df.mode().iloc[0])


print("Range:\n", df.max(numeric_only=True) - df.min(numeric_only=True))
print("Variance:\n", df.var(numeric_only=True))
print("Standard Deviation:\n", df.std(numeric_only=True))
print("IQR:\n", df.quantile(0.75, numeric_only=True) - df.quantile(0.25, numeric_only=True))



print("Skewness:\n", df.skew(numeric_only=True))
print("Kurtosis:\n", df.kurt(numeric_only=True))
print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nInfo:")
print(df.info())

print("\nSummary:\n", df.describe())


df.fillna(df.mean(numeric_only=True), inplace=True)


df.drop_duplicates(inplace=True)


df_num = df.select_dtypes(include=['number'])

print("\nAfter Cleaning:\n", df_num.head())

df_num.hist(figsize=(8,6))
plt.title("Histogram")
plt.show()

if df_num.shape[1] >= 2:
    plt.scatter(df_num.iloc[:,0], df_num.iloc[:,1])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter Plot")
    plt.show()



df_num.boxplot(figsize=(8,6))
plt.title("Box Plot")
plt.show()


corr = df_num.corr()
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()