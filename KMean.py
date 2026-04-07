import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("dataset.csv")




df = df.fillna(df.mean(numeric_only=True))


df_num = df.select_dtypes(include='number')


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print("Cluster Labels:\n", labels[:10])


plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()