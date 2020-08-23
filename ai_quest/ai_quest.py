# %%
import pandas
import numpy
from matplotlib import pyplot
import seaborn
from visualize import visualize_for_continuous, visualize_for_category, visualize_for_category_1d
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
name_continuous_list = numpy.array([
    "accommodates", "bathrooms", "bedrooms", "beds", "latitude", "longitude",
    "number_of_reviews", "review_scores_rating"
])
name_continuous_list = name_continuous_list.reshape(2, 4)
visualize_for_continuous(train, name_continuous_list)

# %%
name_category_list = numpy.array([
    "bed_type", "cancellation_policy", "city", "cleaning_fee",
    "host_has_profile_pic", "host_identity_verified", "instant_bookable",
    "room_type"
])
name_category_list = name_category_list.reshape(4, 2)
visualize_for_category(train, name_category_list)

# %%
name_category_few_list = numpy.array(["host_response_rate", "property_type"])
visualize_for_category_1d(train, name_category_few_list)

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

from sklearn.metrics import mean_squared_error as MSE
import learning

tr = dat[dat["t"] == 1][cols]
learning.k_fold_for_LR_and_RF(tr)
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

model1, model2 = learning.linear_regression_and_random_forest(trainX, y_train)
pred = model1.predict(
    trainX.loc[:, ["number_of_reviews", "review_scores_rating"]]
) + model2.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
model1, model2 = learning.linear_regression_and_random_forest(trainX, y_train)
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

tr = dat[dat["t"] == 1][cols]
learning.k_fold_for_GBR(tr)
learning.k_fold_for_HGBR(tr)

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

# model3 = learning.gradient_boosting_regressor(trainX, y_train)
model4 = learning.hist_gradient_boosting_regressor(trainX, y_train)
# pred = model3.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])
pred = model4.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
# model3 = learning.gradient_boosting_regressor(trainX, y_train)
model4 = learning.hist_gradient_boosting_regressor(trainX, y_train)
# pred = model3.predict(testX.iloc[:, ~testX.columns.str.match("y")])
pred = model4.predict(testX.iloc[:, ~testX.columns.str.match("y")])
pyplot.figure(figsize=(15, 4))
pyplot.plot(pred)

# %%
sample[1] = pred
sample.to_csv("submit01.csv", index=None, header=None)
# %%
