import pandas as pd

df = pd.read_csv(r'C:\Python_Work\機械学習-教科書\スッキリ機械学習\datafiles\iris.csv')
# print(df.head())

# 重複を取り除いた種類列を取得
# print(df["種類"].unique())
# print(type(df["種類"].unique())) # numpy

# データの出現回数をカウント
# print(df["種類"].value_counts())
# print(type(df["種類"].value_counts())) # シリーズ

# print(df.tail())

# 欠損値の確認
# print(df.isnull())

# 列単位で欠損値を確認
# print(df.isnull().any())

# 欠損値の数を求める(Trueなら1, Falseなら0)
# tmp = df.isnull()
# print(tmp.sum())

# 欠損値が1つでもある行を削除
# df2 = df.dropna(how = 'any', axis= 0)
# print(df2.isnull().sum())

df["花弁長さ"] = df["花弁長さ"].fillna(0)
print(df.tail(3))