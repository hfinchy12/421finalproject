# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


from data import load_data, preprocess_x, split_data, match_up_data
from parser import parse
from model import Model


def main():
    args = parse()

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")

    processed_x = preprocess_x(x)
    matched_x, matched_y = match_up_data(x, y)
    train_x, train_y, test_x, test_y = split_data(matched_x, matched_y)

    model = Model(args)  # you can add arguments as needed
    model.fit(train_x, train_y)
    x = load_data("test_x.csv")

    processed_x_test = preprocess_x(x)

    prediction_probs = model.predict_proba(processed_x_test)


if __name__ == "__main__":
    main()
