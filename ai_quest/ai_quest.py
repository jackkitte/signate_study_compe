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
from visualize import (generator_for_1d, generator_for_2d,
                       visualize_for_continuous, visualize_for_continuous_1d,
                       visualize_for_category, visualize_for_category_1d)

seaborn.set(font="IPAexGothic", style="white")
train = pandas.read_csv("./data/train.csv")
train_upper = train.query("y > 185")
train_upper = train_upper.reset_index(drop=True)
test = pandas.read_csv("./data/test.csv")
sample = pandas.read_csv("./data/sample_submit.csv", header=None)

name_continuous_list = numpy.array([
    "accommodates", "bathrooms", "bedrooms", "beds", "number_of_reviews",
    "review_scores_rating"
])
name_category_list = numpy.array([
    "bed_type", "cancellation_policy", "city", "cleaning_fee",
    "host_has_profile_pic", "host_identity_verified", "instant_bookable",
    "room_type"
])
name_category_few_list = numpy.array(
    ["host_response_rate", "property_type", "neighbourhood"])

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
visualize_for_continuous_1d(train, name_continuous_list, "y")

# %%
name_category_list = name_category_list.reshape(4, 2)
visualize_for_category(train, name_category_list, "y")

# %%
visualize_for_category_1d(train, name_category_few_list, "y")

# %%
visualize_for_continuous_1d(train_upper, name_continuous_list, "y")

# %%
name_category_list = name_category_list.reshape(4, 2)
visualize_for_category(train_upper, name_category_list, "y")

# %%
visualize_for_category_1d(train_upper, name_category_few_list, "y")

# %%
generator = generator_for_1d(name_continuous_list)
for name in generator:
    if "latitude" == name or "longitude" == name:
        continue
    else:
        print(f"{name}: \n{sorted(train[name].unique())}\n")

# %%
generator = generator_for_1d(name_category_few_list)
for name in generator:
    if "latitude" == name or "longitude" == name:
        continue
    else:
        print(f"{name}: \n{train[name].unique()}\n")

# %%
generator = generator_for_1d(name_continuous_list)
visualize_for_continuous_1d(train, name_continuous_list, next(generator))
visualize_for_continuous_1d(train, name_continuous_list, next(generator))

# %%
visualize_for_continuous_1d(train, name_continuous_list, next(generator))
visualize_for_continuous_1d(train, name_continuous_list, next(generator))

# %%
train = pandas.read_csv("./data/train.csv")
test = pandas.read_csv("./data/test.csv")
sample = pandas.read_csv("./data/sample_submit.csv", header=None)

train["t"] = 1
test["t"] = 0
dat = pandas.concat([train, test], sort=True).reset_index(drop=True)
dat["review_scores_rating"] = dat["review_scores_rating"].fillna(0)
dat["bathrooms"] = dat["bathrooms"].fillna(0)
dat["bedrooms"] = dat["bathrooms"].fillna(0)
dat["beds"] = dat["bathrooms"].fillna(0)
dat["host_has_profile_pic"] = dat["host_has_profile_pic"].fillna("f")
dat["host_response_rate"] = dat["host_response_rate"].fillna("0%")
dat["neighbourhood"] = dat["neighbourhood"].fillna("None")
"""
cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y"
]
"""
# cols = ["number_of_reviews", "review_scores_rating", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "property_type", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "room_type", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "accommodates", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "bathrooms", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "bedrooms", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "beds", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "city", "y"]
# cols = ["number_of_reviews", "review_scores_rating", "bed_type", "y"]
cols = [
    "number_of_reviews", "review_scores_rating", "accommodates", "beds",
    "latitude", "longitude", "property_type", "room_type", "city",
    "host_response_rate", "y"
]
"""
cols = [
    "number_of_reviews", "review_scores_rating", "cancellation_policy", "y"
]
"""

tr = dat[dat["t"] == 1][cols]
k_fold_for_GBR(tr)
k_fold_for_HGBR(tr)

# %%
"""
cols = [
    "property_type", "cancellation_policy", "room_type", "number_of_reviews",
    "review_scores_rating", "y", "t"
]
"""
cols = [
    "number_of_reviews", "review_scores_rating", "accommodates", "beds",
    "latitude", "longitude", "property_type", "room_type", "city",
    "host_response_rate", "y", "t"
]

tmp = pandas.get_dummies(dat[cols])
trainX = tmp[tmp["t"] == 1]
del trainX["t"]
testX = tmp[tmp["t"] == 0]
del testX["t"]
y_train = tmp[tmp["t"] == 1]["y"]
y_test = tmp[tmp["t"] == 0]["y"]

model4 = hist_gradient_boosting_regressor(trainX, y_train)
pred = model4.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])

p = pandas.DataFrame({"actual": y_train, "pred": pred})
p.plot(figsize=(15, 4))
print("RMSE", MSE(y_train, pred)**0.5)

# %%
model4 = hist_gradient_boosting_regressor(trainX, y_train)
pred = model4.predict(testX.iloc[:, ~testX.columns.str.match("y")])
pyplot.figure(figsize=(15, 4))
pyplot.plot(pred)

# %%
sample[1] = pred
sample.to_csv("./data/submit_HGBR_second.csv", index=None, header=None)
# %%
