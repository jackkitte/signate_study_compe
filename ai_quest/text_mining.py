# %%
import collections
import pandas
import numpy
import re
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
                           value_for_continuous)

seaborn.set(font="IPAexGothic", style="white")
train = pandas.read_csv("./data/train.csv")
train_upper = train.query("y > 185")
train_0_to_25 = train.query("1 <= y < 74")
train_25_to_50 = train.query("74 <= y < 111")
train_50_to_75 = train.query("111 <= y < 185")
train_75_to_100 = train.query("y >= 185")
train_upper = train_upper.reset_index(drop=True)
test = pandas.read_csv("./data/test.csv")
sample = pandas.read_csv("./data/sample_submit.csv", header=None)

name_continuous_list = numpy.array([
    "accommodates", "beds", "number_of_reviews", "review_scores_rating",
    "count_of_amenities", "count_of_description"
])
name_category_list = numpy.array([
    "bed_type", "cancellation_policy", "city", "cleaning_fee", "Gym", "TV",
    "Doorman", "room_type"
])
name_category_few_list = numpy.array(
    ["host_response_rate", "property_type", "neighbourhood"])

print("Data Shapes")
print(
    f"Train: {train.shape} Train_Upper: {train_upper.shape}  Test: {test.shape} Sample: {sample.shape}"
)

# %%
train.describe(include='O')

# %%
generator = generator_for_pandas_tuples(train)

for index in range(10):
    train_row = next(generator)
    count = count_of_description(train_row.description)
    print(count)
    """
    amenitie = amenitie.replace("{", "").replace("}", "")
    amenitie_list = amenitie.split(",")
    print(len(amenitie_list))
    print(amenitie_list)
    """
# %%
train["count_of_amenities"] = train["amenities"].apply(count_of_amenities)
train["count_of_description"] = train["description"].apply(
    count_of_description)
train.describe()
# %%
visualize_for_continuous_1d(train, name_continuous_list, "y")

# %%
train["count_of_amenities"] = train["amenities"].apply(split_of_amenities)
train_generator = generator_for_pandas_rows(train)
list_0_to_25 = []
for index, row in train_generator:
    amenities_generator = generator_for_1d(row["count_of_amenities"])
    for amenitie in amenities_generator:
        try:
            row["amenitie"]
            train.loc[row["id"], amenitie] = "t"
        except:
            train[amenitie] = "f"
            train.loc[row["id"], amenitie] = "t"

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

for index in range(1, 11):
    print(f"{(index-1)*10} ~ {index*10}\n")
    print(f"{most_counter1[(index-1)*10:index*10]}\n")
    print(f"{most_counter2[(index-1)*10:index*10]}\n")
    print(f"{most_counter3[(index-1)*10:index*10]}\n")
    print(f"{most_counter4[(index-1)*10:index*10]}\n")

# %%
dict1 = {}
count_0_30 = most_counter1[0:30]
count_30_70 = most_counter1[30:70]
count_70_over = most_counter1[70:]

generator = generator_for_1d(count_0_30)
dict1 = value_for_1(dict1, generator)
generator = generator_for_1d(count_70_over)
dict1 = value_for_1(dict1, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(-4, 0, 0.1)
gen_continuous = generator_for_1d(continuous)
dict1 = value_for_continuous(dict1, generator, gen_continuous)

print(dict1)
print(len(dict1))

# %%
dict2 = {}
count_0_30 = most_counter2[0:30]
count_30_70 = most_counter2[30:70]
count_70_over = most_counter2[70:]

generator = generator_for_1d(count_0_30)
dict2 = value_for_1(dict2, generator)
generator = generator_for_1d(count_70_over)
dict2 = value_for_1(dict2, generator)
generator = generator_for_1d(count_30_70)
continuous = numpy.arange(-2, 0, 0.05)
gen_continuous = generator_for_1d(continuous)
dict2 = value_for_continuous(dict2, generator, gen_continuous)

print(dict2)
print(len(dict2))

# %%
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

print(dict3)
print(len(dict3))

# %%
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

print(dict4)
print(len(dict4))

# %%
dict_origin = {}
for key, value in dict1.items():
    if dict_origin.get(key):
        dict_origin[key] += value
    else:
        dict_origin[key] = value
for key, value in dict2.items():
    if dict_origin.get(key):
        dict_origin[key] += value
    else:
        dict_origin[key] = value
for key, value in dict3.items():
    if dict_origin.get(key):
        dict_origin[key] += value
    else:
        dict_origin[key] = value
for key, value in dict4.items():
    if dict_origin.get(key):
        dict_origin[key] += value
    else:
        dict_origin[key] = value

# %%
print(dict_origin)
# %%
train["count_of_amenities"] = train["amenities"].apply(count_of_amenities,
                                                       dic=dict_origin)
# %%
train.describe()
# %%

name_continuous_list = numpy.array([
    "accommodates", "beds", "number_of_reviews", "review_scores_rating",
    "count_of_amenities"
])
visualize_for_continuous_1d(train, name_continuous_list, "y")
# %%
name_category_list = numpy.array([
    "bed_type", "cancellation_policy", "city", "cleaning_fee", "Gym", "TV",
    "Doorman", "room_type"
])

name_category_list = name_category_list.reshape(4, 2)
visualize_for_category(train, name_category_list, "y")

# %%
