import numpy
import seaborn
from matplotlib import pyplot


def generator_for_2d(list_for_2d):
    for list_for_1d in list_for_2d:
        for value in list_for_1d:
            yield value


def generator_for_1d(list_for_1d):
    for value in list_for_1d:
        yield value


def visualize_for_continuous(df, name_list, target):
    Color = numpy.array([0.5, 0.5, 0.5]).reshape(1, -1)
    row, column = name_list.shape
    fig, ax = pyplot.subplots(row, column, figsize=(12, 9))
    generator_for_name = generator_for_2d(name_list)
    row_index, column_index = 0, 0
    for name in generator_for_name:
        df.plot.scatter(x=name,
                        y=target,
                        ax=ax[row_index][column_index],
                        c=Color)
        column_index += 1
        if column == column_index:
            row_index += 1
            column_index = 0

    pyplot.tight_layout()


def visualize_for_continuous_1d(df, name_list, target):
    Color = numpy.array([0.5, 0.5, 0.5]).reshape(1, -1)
    column = len(name_list)
    fig, ax = pyplot.subplots(1, column, figsize=(18, 9))
    generator_for_name = generator_for_1d(name_list)
    index = 0
    for name in generator_for_name:
        df.plot.scatter(x=name, y=target, ax=ax[index], c=Color)
        index += 1

    pyplot.tight_layout()


def visualize_for_category(df, name_list, target):
    Color = numpy.array([0.5, 0.5, 0.5]).reshape(1, -1)
    row, column = name_list.shape
    fig, ax = pyplot.subplots(row, column, figsize=(18, 15))
    generator_for_name = generator_for_2d(name_list)
    row_index, column_index = 0, 0
    for name in generator_for_name:
        seaborn.boxplot(x=name,
                        y=target,
                        data=df,
                        ax=ax[row_index][column_index])
        column_index += 1
        if column == column_index:
            row_index += 1
            column_index = 0

    pyplot.tight_layout()


def visualize_for_category_1d(df, name_list, target):
    Color = numpy.array([0.5, 0.5, 0.5]).reshape(1, -1)
    row = len(name_list)
    fig, ax = pyplot.subplots(row, 1, figsize=(18, 15))
    generator_for_name = generator_for_1d(name_list)
    index = 0
    for name in generator_for_name:
        seaborn.boxplot(x=name, y=target, data=df, ax=ax[index])
        index += 1
