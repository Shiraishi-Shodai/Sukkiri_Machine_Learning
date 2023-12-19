import pandas as pd
from sklearn import tree
import joblib

# df = pd.read_csv(r'C:\Python_Work\機械学習-教科書\スッキリ機械学習\datafiles\KvsT.csv')
# # print(df)
# x = df.iloc[:, :-1]
# # print(xcol)

# t = df.iloc[:, -1]
# # print(t)

# model = tree.DecisionTreeClassifier(random_state=0)

# model.fit(x, t)

# joblib.dump(model, r"C:\Python_Work\機械学習-教科書\4章\model")

model = joblib.load(r"C:\Python_Work\機械学習-教科書\4章\model")

# テストデータ１
# test_data = {
#     "身長": [170],
#     "体重": [70],
#     "年代": [20]
# }

# テストデータ２
test_data = {
    "身長": [172, 158],
    "体重": [65, 48],
    "年代": [20, 20]
}

test_df = pd.DataFrame(test_data)
y_pred = model.predict(test_df)
# print(y_pred)
# print(type(y_pred))


score = model.score(x, t)
print(score)