# %%
import pandas
import numpy
import seaborn
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error as MSE

from learning import (linear_regression_and_random_forest,
                      gradient_boosting_regressor,
                      hist_gradient_boosting_regressor, k_fold_for_LR_and_RF,
                      k_fold_for_GBR, k_fold_for_HGBR)
from visualize import visualize_for_continuous, visualize_for_category, visualize_for_category_1d

seaborn.set(font="IPAexGothic", style="white")
train = pandas.read_csv("./data/train.csv")
train_upper = train.query("y > 185")
train_upper = train_upper.reset_index(drop=True)
test = pandas.read_csv("./data/test.csv")
sample = pandas.read_csv("./data/sample_submit.csv", header=None)

print("Data Shapes")
print(
    f"Train: {train.shape} Train_Upper: {train_upper.shape}  Test: {test.shape} Sample: {sample.shape}"
)

# %%
train.head()

# %%
train.describe()

# %%
train.describe(include='O')

# %%
train_upper.head()

# %%
train_upper.describe()

# %%
train_upper.describe(include='O')

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
name_continuous_list = name_continuous_list.reshape(2, 4)
visualize_for_continuous(train_upper, name_continuous_list)

# %%
name_category_list = name_category_list.reshape(4, 2)
visualize_for_category(train_upper, name_category_list)

# %%
visualize_for_category_1d(train_upper, name_category_few_list)

# %%
train = pandas.read_csv("./data/train.csv")
test = pandas.read_csv("./data/test.csv")
sample = pandas.read_csv("./data/sample_submit.csv", header=None)

train["t"] = 1
test["t"] = 0
dat = pandas.concat([train, test], sort=True).reset_index(drop=True)
dat["review_scores_rating"] = dat["review_scores_rating"].fillna(0)

cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y"
]

tr = dat[dat["t"] == 1][cols]
k_fold_for_LR_and_RF(tr)

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

model1, model2 = linear_regression_and_random_forest(trainX, y_train)
pred = model1.predict(
    trainX.loc[:, ["number_of_reviews", "review_scores_rating"]]
) + model2.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
model1, model2 = linear_regression_and_random_forest(trainX, y_train)
pred = model1.predict(
    testX.loc[:,
              ["number_of_reviews", "review_scores_rating"]]) + model2.predict(
                  testX.iloc[:, ~testX.columns.str.match("y")])
pyplot.figure(figsize=(15, 4))
pyplot.plot(pred)

# %%
sample[1] = pred
sample.to_csv("./data/submit01.csv", index=None, header=None)

# %%
train = pandas.read_csv("./data/train.csv")
test = pandas.read_csv("./data/test.csv")
sample = pandas.read_csv("./data/sample_submit.csv", header=None)

train["t"] = 1
test["t"] = 0
dat = pandas.concat([train, test], sort=True).reset_index(drop=True)
dat["review_scores_rating"] = dat["review_scores_rating"].fillna(0)

cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y"
]

tr = dat[dat["t"] == 1][cols]
k_fold_for_GBR(tr)
k_fold_for_HGBR(tr)

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

# model3 = gradient_boosting_regressor(trainX, y_train)
model4 = hist_gradient_boosting_regressor(trainX, y_train)
# pred = model3.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])
pred = model4.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
# model3 = gradient_boosting_regressor(trainX, y_train)
model4 = hist_gradient_boosting_regressor(trainX, y_train)
# pred = model3.predict(testX.iloc[:, ~testX.columns.str.match("y")])
pred = model4.predict(testX.iloc[:, ~testX.columns.str.match("y")])
pyplot.figure(figsize=(15, 4))
pyplot.plot(pred)

# %%
sample[1] = pred
sample.to_csv("./data/submit01.csv", index=None, header=None)
# %%
