import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
feature_names = iris.feature_names
# print(data)
# print(feature_names)

df = pd.DataFrame(data, columns=feature_names)
df["target"] = iris.target
# print(df.head())