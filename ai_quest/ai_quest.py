# %%
import collections
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
from pandas_method import (generator_for_pandas_tuples,
                           generator_for_pandas_rows, count_of_amenities,
                           count_of_description, split_of_amenities,
                           split_of_description, value_for_1,
                           value_for_continuous, continuous_of_amenities,
                           continuous_of_description)

seaborn.set(font="IPAexGothic", style="white")
train = pandas.read_csv("./data/train.csv")
train_upper = train.query("y > 185")
train_upper = train_upper.reset_index(drop=True)
test = pandas.read_csv("./data/test.csv")
sample = pandas.read_csv("./data/sample_submit.csv", header=None)
train_0_to_25 = train.query("1 <= y < 74")
train_25_to_50 = train.query("74 <= y < 111")
train_50_to_75 = train.query("111 <= y < 185")
train_75_to_100 = train.query("y >= 185")

name_continuous_list = numpy.array([
    "accommodates", "beds", "number_of_reviews", "review_scores_rating",
    "count_of_amenities", "count_of_description", "continuous_of_amenities"
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
train["count_of_amenities"] = train["amenities"].apply(count_of_amenities)
train["count_of_description"] = train["description"].apply(
    count_of_description)
# %%
train_0_to_25["count_of_amenities"] = train_0_to_25["amenities"].apply(
    split_of_amenities)
train_25_to_50["count_of_amenities"] = train_25_to_50["amenities"].apply(
    split_of_amenities)
train_50_to_75["count_of_amenities"] = train_50_to_75["amenities"].apply(
    split_of_amenities)
train_75_to_100["count_of_amenities"] = train_75_to_100["amenities"].apply(
    split_of_amenities)

train_0_to_25_generator = generator_for_pandas_tuples(train_0_to_25)
train_25_to_50_generator = generator_for_pandas_tuples(train_25_to_50)
train_50_to_75_generator = generator_for_pandas_tuples(train_50_to_75)
train_75_to_100_generator = generator_for_pandas_tuples(train_75_to_100)

list_0_to_25 = []
list_25_to_50 = []
list_50_to_75 = []
list_75_to_100 = []
for row in train_0_to_25_generator:
    list_0_to_25.extend(row.count_of_amenities)
for row in train_25_to_50_generator:
    list_25_to_50.extend(row.count_of_amenities)
for row in train_50_to_75_generator:
    list_50_to_75.extend(row.count_of_amenities)
for row in train_75_to_100_generator:
    list_75_to_100.extend(row.count_of_amenities)

counter1 = collections.Counter(list_0_to_25)
counter2 = collections.Counter(list_25_to_50)
counter3 = collections.Counter(list_50_to_75)
counter4 = collections.Counter(list_75_to_100)
most_counter1 = counter1.most_common()
most_counter2 = counter2.most_common()
most_counter3 = counter3.most_common()
most_counter4 = counter4.most_common()

dict1 = {}
count_0_30 = most_counter1[0:30]
count_30_70 = most_counter1[30:70]
count_70_over = most_counter1[70:]

generator = generator_for_1d(count_0_30)
dict1 = value_for_1(dict1, generator)
generator = generator_for_1d(count_70_over)
dict1 = value_for_1(dict1, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(-40, 0, 1)
gen_continuous = generator_for_1d(continuous)
dict1 = value_for_continuous(dict1, generator, gen_continuous)

dict2 = {}
count_0_30 = most_counter2[0:30]
count_30_70 = most_counter2[30:70]
count_70_over = most_counter2[70:]

generator = generator_for_1d(count_0_30)
dict2 = value_for_1(dict2, generator)
generator = generator_for_1d(count_70_over)
dict2 = value_for_1(dict2, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(-5, 0, 0.125)
gen_continuous = generator_for_1d(continuous)
dict2 = value_for_continuous(dict2, generator, gen_continuous)

dict3 = {}
count_0_30 = most_counter3[0:30]
count_30_70 = most_counter3[30:70]
count_70_over = most_counter3[70:]

generator = generator_for_1d(count_0_30)
dict3 = value_for_1(dict3, generator)
generator = generator_for_1d(count_70_over)
dict3 = value_for_1(dict3, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(0, 2, 0.05)
gen_continuous = generator_for_1d(continuous)
dict3 = value_for_continuous(dict3, generator, gen_continuous)

dict4 = {}
count_0_30 = most_counter4[0:30]
count_30_70 = most_counter4[30:70]
count_70_over = most_counter4[70:]

generator = generator_for_1d(count_0_30)
dict4 = value_for_1(dict4, generator)
generator = generator_for_1d(count_70_over)
dict4 = value_for_1(dict4, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(0, 4, 0.1)
gen_continuous = generator_for_1d(continuous)
dict4 = value_for_continuous(dict4, generator, gen_continuous)

dict_amenities = {}
for key, value in dict1.items():
    if dict_amenities.get(key):
        dict_amenities[key] += value
    else:
        dict_amenities[key] = value
for key, value in dict2.items():
    if dict_amenities.get(key):
        dict_amenities[key] += value
    else:
        dict_amenities[key] = value
for key, value in dict3.items():
    if dict_amenities.get(key):
        dict_amenities[key] += value
    else:
        dict_amenities[key] = value
for key, value in dict4.items():
    if dict_amenities.get(key):
        dict_amenities[key] += value
    else:
        dict_amenities[key] = value

train_0_to_25["count_of_description"] = train_0_to_25["description"].apply(
    split_of_description)
train_25_to_50["count_of_description"] = train_25_to_50["description"].apply(
    split_of_description)
train_50_to_75["count_of_description"] = train_50_to_75["description"].apply(
    split_of_description)
train_75_to_100["count_of_description"] = train_75_to_100["description"].apply(
    split_of_description)

train_0_to_25_generator = generator_for_pandas_tuples(train_0_to_25)
train_25_to_50_generator = generator_for_pandas_tuples(train_25_to_50)
train_50_to_75_generator = generator_for_pandas_tuples(train_50_to_75)
train_75_to_100_generator = generator_for_pandas_tuples(train_75_to_100)

list_0_to_25 = []
list_25_to_50 = []
list_50_to_75 = []
list_75_to_100 = []
for row in train_0_to_25_generator:
    list_0_to_25.extend(row.count_of_description)
for row in train_25_to_50_generator:
    list_25_to_50.extend(row.count_of_description)
for row in train_50_to_75_generator:
    list_50_to_75.extend(row.count_of_description)
for row in train_75_to_100_generator:
    list_75_to_100.extend(row.count_of_description)

counter1 = collections.Counter(list_0_to_25)
counter2 = collections.Counter(list_25_to_50)
counter3 = collections.Counter(list_50_to_75)
counter4 = collections.Counter(list_75_to_100)
most_counter1 = counter1.most_common()
most_counter2 = counter2.most_common()
most_counter3 = counter3.most_common()
most_counter4 = counter4.most_common()

dict1 = {}
count_0_30 = most_counter1[0:20]
count_30_70 = most_counter1[20:1000]
count_70_over = most_counter1[1000:]

generator = generator_for_1d(count_0_30)
dict1 = value_for_1(dict1, generator)
generator = generator_for_1d(count_70_over)
dict1 = value_for_1(dict1, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(-40, 0, 0.04)
gen_continuous = generator_for_1d(continuous)
dict1 = value_for_continuous(dict1, generator, gen_continuous)

dict2 = {}
count_0_30 = most_counter2[0:20]
count_30_70 = most_counter2[20:1000]
count_70_over = most_counter2[1000:]

generator = generator_for_1d(count_0_30)
dict2 = value_for_1(dict2, generator)
generator = generator_for_1d(count_70_over)
dict2 = value_for_1(dict2, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(-5, 0, 0.005)
gen_continuous = generator_for_1d(continuous)
dict2 = value_for_continuous(dict2, generator, gen_continuous)

dict3 = {}
count_0_30 = most_counter3[0:20]
count_30_70 = most_counter3[20:1000]
count_70_over = most_counter3[1000:]

generator = generator_for_1d(count_0_30)
dict3 = value_for_1(dict3, generator)
generator = generator_for_1d(count_70_over)
dict3 = value_for_1(dict3, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(0, 5, 0.005)
gen_continuous = generator_for_1d(continuous)
dict3 = value_for_continuous(dict3, generator, gen_continuous)

dict4 = {}
count_0_30 = most_counter4[0:20]
count_30_70 = most_counter4[20:1000]
count_70_over = most_counter4[1000:]

generator = generator_for_1d(count_0_30)
dict4 = value_for_1(dict4, generator)
generator = generator_for_1d(count_70_over)
dict4 = value_for_1(dict4, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(0, 40, 0.04)
gen_continuous = generator_for_1d(continuous)
dict4 = value_for_continuous(dict4, generator, gen_continuous)

dict_description = {}
for key, value in dict1.items():
    if dict_description.get(key):
        dict_description[key] += value
    else:
        dict_description[key] = value
for key, value in dict2.items():
    if dict_description.get(key):
        dict_description[key] += value
    else:
        dict_description[key] = value
for key, value in dict3.items():
    if dict_description.get(key):
        dict_description[key] += value
    else:
        dict_description[key] = value
for key, value in dict4.items():
    if dict_description.get(key):
        dict_description[key] += value
    else:
        dict_description[key] = value

# %%
train["continuous_of_amenities"] = train["amenities"].apply(
    continuous_of_amenities, dic=dict_amenities)
train["continuous_of_description"] = train["description"].apply(
    continuous_of_description, dic=dict_description)

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

train["count_of_amenities"] = train["amenities"].apply(count_of_amenities)
train["count_of_description"] = train["description"].apply(
    count_of_description)
train["continuous_of_amenities"] = train["amenities"].apply(
    continuous_of_amenities, dic=dict_amenities)
train["continuous_of_description"] = train["description"].apply(
    continuous_of_description, dic=dict_description)

test["count_of_amenities"] = test["amenities"].apply(count_of_amenities)
test["count_of_description"] = test["description"].apply(count_of_description)
test["continuous_of_amenities"] = train["amenities"].apply(
    continuous_of_amenities, dic=dict_amenities)
test["continuous_of_description"] = test["description"].apply(
    continuous_of_description, dic=dict_description)

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

cols = [
    "number_of_reviews", "review_scores_rating", "accommodates", "beds",
    "latitude", "longitude", "property_type", "room_type", "city",
    "host_response_rate", "count_of_description", "count_of_amenities",
    "continuous_of_amenities", "continuous_of_description", "y"
]

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
    "host_response_rate", "count_of_description", "continuous_of_amenities",
    "continuous_of_description", "y", "t"
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
sample.to_csv("./data/submit_HGBR_7.csv", index=None, header=None)
# %%
