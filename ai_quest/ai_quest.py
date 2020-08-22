# %%
import pandas
import numpy
from matplotlib import pyplot
import seaborn
seaborn.set(font="IPAexGothic", style="white")

train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")
sample = pandas.read_csv("sample_submit.csv", header=None)
print("Data Shapes")
print("Train:", train.shape, "Test:", test.shape, "Sample:", sample.shape)

# %%
train.head()
# %%
train.describe()
# %%
train.describe(include='O')

# %%
train_uppder = train.query("y > 185")
train_uppder = train_uppder.reset_index(drop=True)
train_uppder.head()
# %%
train_uppder.describe()

# %%
train_uppder.describe(include='O')

# %%
Color = numpy.array([0.5, 0.5, 0.5]).reshape(1, -1)
fig, ax = pyplot.subplots(2, 4, figsize=(12, 9))
train_uppder.plot.scatter(x="accommodates", y="y", ax=ax[0][0], c=Color)
train_uppder.plot.scatter(x="bathrooms", y="y", ax=ax[0][1], c=Color)
train_uppder.plot.scatter(x="bedrooms", y="y", ax=ax[0][2], c=Color)
train_uppder.plot.scatter(x="beds", y="y", ax=ax[0][3], c=Color)
train_uppder.plot.scatter(x="latitude", y="y", ax=ax[1][0], c=Color)
train_uppder.plot.scatter(x="longitude", y="y", ax=ax[1][1], c=Color)
train_uppder.plot.scatter(x="number_of_reviews", y="y", ax=ax[1][2], c=Color)
train_uppder.plot.scatter(x="review_scores_rating",
                          y="y",
                          ax=ax[1][3],
                          c=Color)
pyplot.tight_layout()

# %%
fig, ax = pyplot.subplots(4, 2, figsize=(18, 15))
seaborn.boxplot(x="bed_type", y="y", data=train_uppder, ax=ax[0][0])
seaborn.boxplot(x="cancellation_policy", y="y", data=train_uppder, ax=ax[0][1])
seaborn.boxplot(x="city", y="y", data=train_uppder, ax=ax[1][0])
seaborn.boxplot(x="cleaning_fee", y="y", data=train_uppder, ax=ax[1][1])
seaborn.boxplot(x="host_has_profile_pic",
                y="y",
                data=train_uppder,
                ax=ax[2][0])
seaborn.boxplot(x="host_identity_verified",
                y="y",
                data=train_uppder,
                ax=ax[2][1])
seaborn.boxplot(x="instant_bookable", y="y", data=train_uppder, ax=ax[3][0])
seaborn.boxplot(x="room_type", y="y", data=train_uppder, ax=ax[3][1])
pyplot.tight_layout()

# %%
fig, ax = pyplot.subplots(2, 1, figsize=(30, 15))
seaborn.boxplot(x="host_response_rate", y="y", data=train_uppder, ax=ax[0])
seaborn.boxplot(x="property_type", y="y", data=train_uppder, ax=ax[1])
pyplot.tight_layout()
# %%
Color = numpy.array([0.5, 0.5, 0.5]).reshape(1, -1)
fig, ax = pyplot.subplots(2, 4, figsize=(12, 9))
train.plot.scatter(x="accommodates", y="y", ax=ax[0][0], c=Color)
train.plot.scatter(x="bathrooms", y="y", ax=ax[0][1], c=Color)
train.plot.scatter(x="bedrooms", y="y", ax=ax[0][2], c=Color)
train.plot.scatter(x="beds", y="y", ax=ax[0][3], c=Color)
train.plot.scatter(x="latitude", y="y", ax=ax[1][0], c=Color)
train.plot.scatter(x="longitude", y="y", ax=ax[1][1], c=Color)
train.plot.scatter(x="number_of_reviews", y="y", ax=ax[1][2], c=Color)
train.plot.scatter(x="review_scores_rating", y="y", ax=ax[1][3], c=Color)
pyplot.tight_layout()

# %%
fig, ax = pyplot.subplots(4, 2, figsize=(18, 15))
seaborn.boxplot(x="bed_type", y="y", data=train, ax=ax[0][0])
seaborn.boxplot(x="cancellation_policy", y="y", data=train, ax=ax[0][1])
seaborn.boxplot(x="city", y="y", data=train, ax=ax[1][0])
seaborn.boxplot(x="cleaning_fee", y="y", data=train, ax=ax[1][1])
seaborn.boxplot(x="host_has_profile_pic", y="y", data=train, ax=ax[2][0])
seaborn.boxplot(x="host_identity_verified", y="y", data=train, ax=ax[2][1])
seaborn.boxplot(x="instant_bookable", y="y", data=train, ax=ax[3][0])
seaborn.boxplot(x="room_type", y="y", data=train, ax=ax[3][1])
pyplot.tight_layout()

# %%
fig, ax = pyplot.subplots(2, 1, figsize=(30, 15))
seaborn.boxplot(x="host_response_rate", y="y", data=train, ax=ax[0])
seaborn.boxplot(x="property_type", y="y", data=train, ax=ax[1])
pyplot.tight_layout()
# %%
train = pandas.read_csv("./train.csv")
test = pandas.read_csv("./test.csv")
sample = pandas.read_csv("./sample_submit.csv", header=None)

train["t"] = 1
test["t"] = 0
dat = pandas.concat([train, test], sort=True).reset_index(drop=True)
dat["review_scores_rating"] = dat["review_scores_rating"].fillna(0)

cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y"
]

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF


def learning(trainX, y_train):
    model1 = LR()
    model2 = RF(n_estimators=100, max_depth=4, random_state=777)
    model1.fit(trainX.loc[:, ["number_of_reviews", "review_scores_rating"]],
               y_train)
    pred = model1.predict(
        trainX.loc[:, ["number_of_reviews", "review_scores_rating"]])

    pred_sub = y_train - pred
    model2.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], pred_sub)
    return model1, model2


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

    pred_train = model1.predict(
        trainX.loc[:, ["number_of_reviews", "review_scores_rating"]]
    ) + model2.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])
    pred_test = model1.predict(
        testX.loc[:, ["number_of_reviews", "review_scores_rating"]]
    ) + model2.predict(testX.iloc[:, ~testX.columns.str.match("y")])

    print("TRAIN:",
          MSE(y_train, pred_train)**0.5, "VARIDATE",
          MSE(y_test, pred_test)**0.5)
    trains.append(MSE(y_train, pred_train)**0.5)
    tests.append(MSE(y_test, pred_test)**0.5)
print("AVG")
print(numpy.array(trains).mean(), numpy.array(tests).mean())

# %%
cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y", "t"
]
tmp = pandas.get_dummies(dat[cols])
trainX = tmp[tmp["t"] == 1]
del trainX["t"]
testX = tmp[tmp["t"] == 0]
del testX["t"]
y_train = tmp[tmp["t"] == 1]["y"]
y_test = tmp[tmp["t"] == 0]["y"]

model1, model2 = learning(trainX, y_train)
pred = model1.predict(
    trainX.loc[:, ["number_of_reviews", "review_scores_rating"]]
) + model2.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
model1, model2 = learning(trainX, y_train)
pred = model1.predict(
    testX.loc[:,
              ["number_of_reviews", "review_scores_rating"]]) + model2.predict(
                  testX.iloc[:, ~testX.columns.str.match("y")])
pyplot.figure(figsize=(15, 4))
pyplot.plot(pred)

# %%
sample[1] = pred
sample.to_csv("submit01.csv", index=None, header=None)


# %%
def learning_only(trainX, y_train):
    model2 = RF(n_estimators=100, max_depth=4, random_state=777)
    model2.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], y_train)
    return model2


train = pandas.read_csv("./train.csv")
test = pandas.read_csv("./test.csv")
sample = pandas.read_csv("./sample_submit.csv", header=None)

train["t"] = 1
test["t"] = 0
dat = pandas.concat([train, test], sort=True).reset_index(drop=True)
dat["review_scores_rating"] = dat["review_scores_rating"].fillna(0)

cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y"
]

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

    model2 = learning_only(trainX, y_train)

    pred_train = model2.predict(trainX.iloc[:, ~testX.columns.str.match("y")])
    pred_test = model2.predict(testX.iloc[:, ~testX.columns.str.match("y")])

    print("TRAIN:",
          MSE(y_train, pred_train)**0.5, "VARIDATE",
          MSE(y_test, pred_test)**0.5)
    trains.append(MSE(y_train, pred_train)**0.5)
    tests.append(MSE(y_test, pred_test)**0.5)
print("AVG")
print(numpy.array(trains).mean(), numpy.array(tests).mean())
# %%
cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y", "t"
]
tmp = pandas.get_dummies(dat[cols])
tmp_upper = tmp.query("y > 185")
tmp_down = tmp.query("y <= 185")

trainX = tmp_down[tmp_down["t"] == 1]
trainX_upper = tmp_upper[tmp_upper["t"] == 1]
del trainX["t"]
del trainX_upper["t"]
testX = tmp[tmp["t"] == 0]
del testX["t"]
y_train = tmp_down[tmp_down["t"] == 1]["y"]
y_train_upper = tmp_upper[tmp_upper["t"] == 1]["y"]
y_test = tmp[tmp["t"] == 0]["y"]

model2 = learning_only(trainX, y_train)
model2_upper = learning_only(trainX_upper, y_train_upper)
pred = (
    model2.predict(trainX.iloc[:, ~trainX.columns.str.match("y")]) +
    model2_upper.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])) / 2

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
model2 = learning_only(trainX, y_train)
model2_upper = learning_only(trainX_upper, y_train_upper)
pred = (model2.predict(testX.iloc[:, ~testX.columns.str.match("y")]) +
        model2_upper.predict(testX.iloc[:, ~testX.columns.str.match("y")])) / 2

pyplot.figure(figsize=(15, 4))
pyplot.plot(pred)

# %%
sample[1] = pred
sample.to_csv("submit01.csv", index=None, header=None)
# %%
