import pandas as newpd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

newdf = newpd.read_csv("data.csv")

# Use numeric data
newdf = newdf.select_dtypes(include=['number'])

newX = newdf.iloc[:, :-1]
newy = newdf.iloc[:, -1]

newX_train, newX_test, newy_train, newy_test = train_test_split(newX, newy, test_size=0.3)

newmodel = KNeighborsClassifier(n_neighbors=3)
newmodel.fit(newX_train, newy_train)

print("Accuracy:", newmodel.score(newX_test, newy_test))