import numpy
import pandas
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.svm import SVR, NuSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def linear_regression_and_random_forest(trainX, y_train):
    model1 = LR()
    model2 = RF(n_estimators=100, max_depth=4, random_state=777)
    model1.fit(trainX.loc[:, ["number_of_reviews", "review_scores_rating"]],
               y_train)
    pred = model1.predict(
        trainX.loc[:, ["number_of_reviews", "review_scores_rating"]])

    pred_sub = y_train - pred
    model2.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], pred_sub)
    return model1, model2


def k_fold_for_LR_and_RF(train_data):
    kf = KFold(n_splits=5, random_state=777)

    trains = []
    tests = []
    for train_index, test_index in kf.split(train_data):
        train_data.loc[train_index, "tt"] = 1
        train_data.loc[test_index, "tt"] = 0
        train_data["tt"] = train_data["tt"].astype(numpy.int)
        tmp = pandas.get_dummies(train_data)

        trainX = tmp[tmp["tt"] == 1]
        del trainX["tt"]
        testX = tmp[tmp["tt"] == 0]
        del testX["tt"]
        y_train = tmp[tmp["tt"] == 1]["y"]
        y_test = tmp[tmp["tt"] == 0]["y"]

        model1, model2 = linear_regression_and_random_forest(trainX, y_train)

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


def svr(trainX, y_train):
    model = make_pipeline(StandardScaler(),
                          SVR(kernel="poly", C=1, epsilon=0.1, max_iter=500))
    model.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], y_train)

    return model


def nu_svr(trainX, y_train):
    model = make_pipeline(StandardScaler(),
                          NuSVR(kernel="poly", C=1, nu=0.5, max_iter=20))

    model.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], y_train)

    return model


def origin_gradient_boosting_regressor(trainX, y_train):
    model = GBR(n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=777,
                loss='ls')
    model.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], y_train)
    return model


def gradient_boosting_regressor(trainX, y_train):
    model = GBR(n_estimators=300,
                learning_rate=0.1,
                max_depth=8,
                random_state=777,
                loss='ls')
    model.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], y_train)
    return model


def origin_hist_gradient_boosting_regressor(trainX, y_train):
    model = HGBR(max_iter=100,
                 learning_rate=0.5,
                 max_depth=10,
                 random_state=777,
                 loss='least_squares')
    model.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], y_train)
    return model


def hist_gradient_boosting_regressor(trainX, y_train):
    model = HGBR(max_iter=200,
                 learning_rate=0.1,
                 max_leaf_nodes=150,
                 random_state=777,
                 loss='least_squares')
    model.fit(trainX.iloc[:, ~trainX.columns.str.match("y")], y_train)
    return model


def k_fold_for_svr(train_data):
    kf = KFold(n_splits=5, random_state=777)

    trains = []
    tests = []
    for train_index, test_index in kf.split(train_data):
        train_data.loc[train_index, "tt"] = 1
        train_data.loc[test_index, "tt"] = 0
        train_data["tt"] = train_data["tt"].astype(numpy.int)
        tmp = pandas.get_dummies(train_data)

        trainX = tmp[tmp["tt"] == 1]
        del trainX["tt"]
        testX = tmp[tmp["tt"] == 0]
        del testX["tt"]
        y_train = tmp[tmp["tt"] == 1]["y"]
        y_test = tmp[tmp["tt"] == 0]["y"]

        model = svr(trainX, y_train)

        pred_train = model.predict(trainX.iloc[:,
                                               ~trainX.columns.str.match("y")])
        pred_test = model.predict(testX.iloc[:, ~testX.columns.str.match("y")])

        print("TRAIN:",
              MSE(y_train, pred_train)**0.5, "VARIDATE",
              MSE(y_test, pred_test)**0.5)
        trains.append(MSE(y_train, pred_train)**0.5)
        tests.append(MSE(y_test, pred_test)**0.5)
    print("AVG")
    print(numpy.array(trains).mean(), numpy.array(tests).mean())


def k_fold_for_nu_svr(train_data):
    kf = KFold(n_splits=10, random_state=777)

    trains = []
    tests = []
    for train_index, test_index in kf.split(train_data):
        train_data.loc[train_index, "tt"] = 1
        train_data.loc[test_index, "tt"] = 0
        train_data["tt"] = train_data["tt"].astype(numpy.int)
        tmp = pandas.get_dummies(train_data)

        trainX = tmp[tmp["tt"] == 1]
        del trainX["tt"]
        testX = tmp[tmp["tt"] == 0]
        del testX["tt"]
        y_train = tmp[tmp["tt"] == 1]["y"]
        y_test = tmp[tmp["tt"] == 0]["y"]

        model = nu_svr(trainX, y_train)

        pred_train = model.predict(trainX.iloc[:,
                                               ~trainX.columns.str.match("y")])
        pred_test = model.predict(testX.iloc[:, ~testX.columns.str.match("y")])

        print("TRAIN:",
              MSE(y_train, pred_train)**0.5, "VARIDATE",
              MSE(y_test, pred_test)**0.5)
        trains.append(MSE(y_train, pred_train)**0.5)
        tests.append(MSE(y_test, pred_test)**0.5)
    print("AVG")
    print(numpy.array(trains).mean(), numpy.array(tests).mean())


def k_fold_for_origin_GBR(train_data):
    kf = KFold(n_splits=5, random_state=777)

    trains = []
    tests = []
    for train_index, test_index in kf.split(train_data):
        train_data.loc[train_index, "tt"] = 1
        train_data.loc[test_index, "tt"] = 0
        train_data["tt"] = train_data["tt"].astype(numpy.int)
        tmp = pandas.get_dummies(train_data)

        trainX = tmp[tmp["tt"] == 1]
        del trainX["tt"]
        testX = tmp[tmp["tt"] == 0]
        del testX["tt"]
        y_train = tmp[tmp["tt"] == 1]["y"]
        y_test = tmp[tmp["tt"] == 0]["y"]

        model = origin_gradient_boosting_regressor(trainX, y_train)

        pred_train = model.predict(trainX.iloc[:,
                                               ~trainX.columns.str.match("y")])
        pred_test = model.predict(testX.iloc[:, ~testX.columns.str.match("y")])

        print("TRAIN:",
              MSE(y_train, pred_train)**0.5, "VARIDATE",
              MSE(y_test, pred_test)**0.5)
        trains.append(MSE(y_train, pred_train)**0.5)
        tests.append(MSE(y_test, pred_test)**0.5)
    print("AVG")
    print(numpy.array(trains).mean(), numpy.array(tests).mean())


def k_fold_for_GBR(train_data):
    kf = KFold(n_splits=5, random_state=777)

    trains = []
    tests = []
    for train_index, test_index in kf.split(train_data):
        train_data.loc[train_index, "tt"] = 1
        train_data.loc[test_index, "tt"] = 0
        train_data["tt"] = train_data["tt"].astype(numpy.int)
        tmp = pandas.get_dummies(train_data)

        trainX = tmp[tmp["tt"] == 1]
        del trainX["tt"]
        testX = tmp[tmp["tt"] == 0]
        del testX["tt"]
        y_train = tmp[tmp["tt"] == 1]["y"]
        y_test = tmp[tmp["tt"] == 0]["y"]

        model = gradient_boosting_regressor(trainX, y_train)

        pred_train = model.predict(trainX.iloc[:,
                                               ~trainX.columns.str.match("y")])
        pred_test = model.predict(testX.iloc[:, ~testX.columns.str.match("y")])

        print("TRAIN:",
              MSE(y_train, pred_train)**0.5, "VARIDATE",
              MSE(y_test, pred_test)**0.5)
        trains.append(MSE(y_train, pred_train)**0.5)
        tests.append(MSE(y_test, pred_test)**0.5)
    print("AVG")
    print(numpy.array(trains).mean(), numpy.array(tests).mean())


def k_fold_for_origin_HGBR(train_data):
    kf = KFold(n_splits=5, random_state=777)

    trains = []
    tests = []
    for train_index, test_index in kf.split(train_data):
        train_data.loc[train_index, "tt"] = 1
        train_data.loc[test_index, "tt"] = 0
        train_data["tt"] = train_data["tt"].astype(numpy.int)
        tmp = pandas.get_dummies(train_data)

        trainX = tmp[tmp["tt"] == 1]
        del trainX["tt"]
        testX = tmp[tmp["tt"] == 0]
        del testX["tt"]
        y_train = tmp[tmp["tt"] == 1]["y"]
        y_test = tmp[tmp["tt"] == 0]["y"]

        model = origin_hist_gradient_boosting_regressor(trainX, y_train)

        pred_train = model.predict(trainX.iloc[:,
                                               ~trainX.columns.str.match("y")])
        pred_test = model.predict(testX.iloc[:, ~testX.columns.str.match("y")])

        print("TRAIN:",
              MSE(y_train, pred_train)**0.5, "VARIDATE",
              MSE(y_test, pred_test)**0.5)
        trains.append(MSE(y_train, pred_train)**0.5)
        tests.append(MSE(y_test, pred_test)**0.5)
    print("AVG")
    print(numpy.array(trains).mean(), numpy.array(tests).mean())


def k_fold_for_HGBR(train_data):
    kf = KFold(n_splits=5, random_state=777)

    trains = []
    tests = []
    for train_index, test_index in kf.split(train_data):
        train_data.loc[train_index, "tt"] = 1
        train_data.loc[test_index, "tt"] = 0
        train_data["tt"] = train_data["tt"].astype(numpy.int)
        tmp = pandas.get_dummies(train_data)

        trainX = tmp[tmp["tt"] == 1]
        del trainX["tt"]
        testX = tmp[tmp["tt"] == 0]
        del testX["tt"]
        y_train = tmp[tmp["tt"] == 1]["y"]
        y_test = tmp[tmp["tt"] == 0]["y"]

        model = hist_gradient_boosting_regressor(trainX, y_train)

        pred_train = model.predict(trainX.iloc[:,
                                               ~trainX.columns.str.match("y")])
        pred_test = model.predict(testX.iloc[:, ~testX.columns.str.match("y")])

        print("TRAIN:",
              MSE(y_train, pred_train)**0.5, "VARIDATE",
              MSE(y_test, pred_test)**0.5)
        trains.append(MSE(y_train, pred_train)**0.5)
        tests.append(MSE(y_test, pred_test)**0.5)
    print("AVG")
    print(numpy.array(trains).mean(), numpy.array(tests).mean())
