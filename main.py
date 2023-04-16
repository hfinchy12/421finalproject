# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools
import os

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import load_data, preprocess_x, split_data, match_up_data
from parser import parse
from model import Model
import time

TRAIN_X_FILE = "train_x.csv"
TRAIN_Y_FILE = "train_y.csv"
TEST_X_FILE = "test_x.csv"
PROCESSED_TRAIN_X_FILE = "processed_train_x.csv"
PROCESSED_TEST_X_FILE = "processed_test_x.csv"
SUBMISSION_FILE = "submission.csv"


def main():
    args = parse()

    x, y, submission_x = None, None, None

    if args.use_preprocessed:
        print(
            f"Loading preprocessed data located in {os.path.join(args.data_path, PROCESSED_TRAIN_X_FILE)}.")

        processed_x = load_data(os.path.join(args.data_path, PROCESSED_TRAIN_X_FILE))
        y = load_data(os.path.join(args.data_path, TRAIN_Y_FILE))

        submission_x = load_data(os.path.join(
            args.data_path, PROCESSED_TEST_X_FILE))

    else:
        x = load_data(os.path.join(args.data_path, "train_x.csv"))
        y = load_data(os.path.join(args.data_path, "train_y.csv"))

        save_path_train = None
        save_path_test = None
        if args.save_processed_data != None:
            save_path_train = os.path.join(
                args.data_path, PROCESSED_TRAIN_X_FILE)
            save_path_test = os.path.join(
                args.data_path, PROCESSED_TEST_X_FILE)
            print(
                f"Processed data will be saved to {save_path_train} and {save_path_test}")

        start_time = time.time()
        print(
            f"Processing data located in {os.path.join(args.data_path, 'train_x.csv')}.")
        processed_x = preprocess_x(
            x, estimateData=False, savePath=save_path_train)
        print(
            f"Processing data located in {os.path.join(args.data_path, 'test_x.csv')}.")
        submission_x = preprocess_x(load_data(os.path.join(
            args.data_path, 'test_x.csv')), estimateData=True, savePath=save_path_test)
        end_time = time.time()
        print(f"Processed the data in {end_time - start_time} seconds.")



    matched_x, matched_y = match_up_data(processed_x, y)
    train_x, train_y, test_x, test_y = split_data(matched_x, matched_y)

    print(f"Beginning model training.")
    # TODO: pass in arguments from commandline
    model = Model()
    model.fit(train_x, train_y, test_x, test_y)

    model.submit(submission_x, os.path.join(args.data_path, SUBMISSION_FILE))

    model.save()


if __name__ == "__main__":
    main()
