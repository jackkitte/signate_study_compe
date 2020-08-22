# %%
import pandas
import numpy
from matplotlib import pyplot
import seaborn
seaborn.set(font="IPAexGothic", style="white")
import matplotlib
# %%
train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")
sample = pandas.read_csv("sample.csv", header=None)
print("Data Shapes")
print("Train:", train.shape, "Test:", test.shape, "Sample:", sample.shape)

# %%
train.index = pandas.to_datetime(train.datetime)
train.head()
# %%
train.describe()
# %%
train.describe(include='O')
# %%
train["payday"] = train["payday"].fillna(0)
train["precipitation"] = train["precipitation"].apply(
    lambda x: -1 if x == "--" else float(x))
train["event"] = train["event"].fillna("なし")
train["remarks"] = train["remarks"].fillna("なし")
train["month"] = train["datetime"].apply(lambda x: int(x.split("-")[1]))

# %%
train.y.plot(figsize=(15, 4))
# %%
Color = numpy.array([0.5, 0.5, 0.5]).reshape(1, -1)
fig, ax = pyplot.subplots(2, 3, figsize=(12, 9))
train.plot.scatter(x="soldout", y="y", ax=ax[0][0], c=Color)
train.plot.scatter(x="kcal", y="y", ax=ax[0][1], c=Color)
train.plot.scatter(x="precipitation", y="y", ax=ax[0][2], c=Color)
train.plot.scatter(x="payday", y="y", ax=ax[1][0], c=Color)
train.plot.scatter(x="temperature", y="y", ax=ax[1][1], c=Color)
train.plot.scatter(x="month", y="y", ax=ax[1][2], c=Color)
pyplot.tight_layout()
# %%
fig, ax = pyplot.subplots(2, 2, figsize=(12, 7))
seaborn.boxplot(x="week", y="y", data=train, ax=ax[0][0])
seaborn.boxplot(x="weather", y="y", data=train, ax=ax[0][1])
seaborn.boxplot(x="remarks", y="y", data=train, ax=ax[1][0])
ax[1][0].set_xticklabels(ax[1][0].get_xticklabels(), rotation=30)
seaborn.boxplot(x="event", y="y", data=train, ax=ax[1][1])
pyplot.tight_layout()
# %%
train[train["remarks"] != "お楽しみメニュー"]["y"].plot(figsize=(15, 4))
# %%
train["fun"] = train["remarks"].apply(lambda x: 1 if x == "お楽しみメニュー" else 0)
seaborn.boxplot(x="fun", y="y", data=train)
# %%
train[train["remarks"] == "お楽しみメニュー"]["y"].plot(figsize=(15, 4))
# %%
train[train["remarks"] == "お楽しみメニュー"]
# %%
train["curry"] = train["name"].apply(lambda x: 1 if x.find("カレー") >= 0 else 0)
seaborn.boxplot(x="curry", y="y", data=train)

# %%
train = pandas.read_csv("./train.csv")
test = pandas.read_csv("./test.csv")
sample = pandas.read_csv("./sample.csv", header=None)

train["t"] = 1
test["t"] = 0
dat = pandas.concat([train, test], sort=True).reset_index(drop=True)

dat.index = pandas.to_datetime(dat["datetime"])
dat = dat["2014-05-01":]
dat = dat.reset_index(drop=True)

dat["days"] = dat.index
dat["precipitation"] = dat["precipitation"].apply(
    lambda x: -1 if x == "--" else x).astype(numpy.float)
dat["fun"] = dat["remarks"].apply(lambda x: 1 if x == "お楽しみメニュー" else 0)
dat["curry"] = dat["name"].apply(lambda x: 1 if x.find("カレー") >= 0 else 0)

cols = ["precipitation", "weather", "days", "fun", "curry", "y"]

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF


def learning(trainX, y_train):
    model1 = LR()
    model2 = RF(n_estimators=100, max_depth=4, random_state=777)
    model1.fit(trainX["days"].values.reshape(-1, 1), y_train)
    pred = model1.predict(trainX["days"].values.reshape(-1, 1))

    pred_sub = y_train - pred
    model2.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], pred_sub)
    return model1, model2


# %%
kf = KFold(n_splits=5, random_state=777)
tr = dat[dat["t"] == 1][cols]

trains = []
tests = []
for train_index, test_index in kf.split(tr):
    tr.loc[train_index, "tt"] = 1
    tr.loc[test_index, "tt"] = 0
    tr["tt"] = tr["tt"].astype(numpy.int)
    tmp = pandas.get_dummies(tr)

    trainX = tmp[tmp["tt"] == 1]
    del trainX["tt"]
    testX = tmp[tmp["tt"] == 0]
    del testX["tt"]
    y_train = tmp[tmp["tt"] == 1]["y"]
    y_test = tmp[tmp["tt"] == 0]["y"]

    model1, model2 = learning(trainX, y_train)

    pred_train = model1.predict(trainX["days"].values.reshape(
        -1, 1)) + model2.predict(trainX.iloc[:,
                                             ~trainX.columns.str.match("y")])
    pred_test = model1.predict(testX["days"].values.reshape(
        -1, 1)) + model2.predict(testX.iloc[:, ~testX.columns.str.match("y")])

    print("TRAIN:",
          MSE(y_train, pred_train)**0.5, "VARIDATE",
          MSE(y_test, pred_test)**0.5)
    trains.append(MSE(y_train, pred_train)**0.5)
    tests.append(MSE(y_test, pred_test)**0.5)
print("AVG")
print(numpy.array(trains).mean(), numpy.array(tests).mean())
# %%
cols = ["precipitation", "weather", "days", "fun", "curry", "y", "t"]
tmp = pandas.get_dummies(dat[cols])
trainX = tmp[tmp["t"] == 1]
del trainX["t"]
testX = tmp[tmp["t"] == 0]
del testX["t"]
y_train = tmp[tmp["t"] == 1]["y"]
y_test = tmp[tmp["t"] == 0]["y"]

model1, model2 = learning(trainX, y_train)
pred = model1.predict(trainX["days"].values.reshape(-1, 1)) + model2.predict(
    trainX.iloc[:, ~trainX.columns.str.match("y")])

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
model1, model2 = learning(trainX, y_train)
pred = model1.predict(testX["days"].values.reshape(-1, 1)) + model2.predict(
    testX.iloc[:, ~testX.columns.str.match("y")])
pyplot.figure(figsize=(15, 4))
pyplot.plot(pred)
# %%
sample[1] = pred
sample.to_csv("submit01.csv", index=None, header=None)

# %%
