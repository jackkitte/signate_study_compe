import re


def generator_for_pandas_tuples(df):
    for row in df.itertuples():
        yield row


def generator_for_pandas_rows(df):
    for index, row in df.iterrows():
        yield index, row


def continuous_of_amenities(text, dic):
    text = text.replace("{", "").replace("}", "")
    text_list = text.split(",")
    sum_of_word = 0
    for word in text_list:
        sum_of_word += dic[word]

    return sum_of_word


def count_of_amenities(text):
    text = text.replace("{", "").replace("}", "")
    text_list = text.split(",")

    return len(text_list)


def count_of_description(text):
    text_sub = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text_sub = re.sub(r'[!-@[-`{-~]', "", text_sub)

    text_list = text.split()

    return len(text_list)


def split_of_amenities(text):
    text = text.replace("{", "").replace("}", "")
    text_list = text.split(",")

    return text_list


def split_of_description(text):
    text_sub = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text_sub = re.sub(r'[!-@[-`{-~]', "", text_sub)

    text_list = text.split()

    return text_list


def value_for_1(dic, generator):
    for word, _ in generator:
        dic[word] = 1

    return dic


def value_for_continuous(dic, generator, continuous):
    for word, _ in generator:
        dic[word] = next(continuous)

    return dic
