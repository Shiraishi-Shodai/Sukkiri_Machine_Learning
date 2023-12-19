import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle


#  -------前処理-------


# データフレームを読み込む
df = pd.read_csv("cinema.csv")

# 欠損値の確認
print(df.isnull().any(axis=0))


# 欠損値を平均値で埋める
df2 = df.fillna(df.mean())
# 欠損値の確認
print(df2.isnull().any(axis=0))

fig = plt.figure()

# sub1 = fig.add_subplot(2, 2, 1)
# sub1.scatter(x=df2['SNS1'], y=df2['sales'])

sub2 = fig.add_subplot(2, 2, 2)
sub2.scatter(x=df2['SNS2'], y=df2['sales'])

# sub3 = fig.add_subplot(2, 2, 3)
# sub3.scatter(x=df2['actor'], y=df2['sales'])

# sub4 = fig.add_subplot(2, 2, 4)
# sub4.scatter(x=df2['original'], y=df2['sales'])

# 外れ値のインデックスを検索
no = df2[(df2['SNS2'] > 1000) & (df2['sales'] < 8500)].index
print(f'外れ値のインデックス{no}')
# 外れ値がインデックスを行方向に削除
df3 = df2.drop(no, axis= 0)
# print(df3.shape)


#  -------学習-------

col = ["SNS1", "SNS2", "actor", "actor", "original"]
x = df3[col] #特徴量の取り出し

t = df3['sales'] #正解データの取り出し

# 訓練データとテストデータを分割
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# 学習

# 平均二乗誤差
model = LinearRegression()
model.fit(x_train, y_train)


#  -------評価-------

print(f'決定係数(大きいほど予測値と実測値との残差が小さい){model.score(x_test, y_test)}')
# x_testを読み込みyを予測
pred = model.predict(x_test)

# 平均絶対誤差
ans = mean_absolute_error(y_pred=pred, y_true=y_test)
print(f'平均絶対誤差{ans}')

# モデルをファイルに保存
with open('cinema.pkl', 'wb') as f:
    pickle.dump(model, f)
    
    
# グラフの保存
plt.savefig("test.png")

