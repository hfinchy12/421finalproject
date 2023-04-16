import argparse


def parse():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "-d", "--data_path", type=str, default="data", help="path to your data directory"
    )

    parser.add_argument(
        "-p", "--use_preprocessed", action="store_true", help="use preprocessed data"
    )

    parser.add_argument(
        "--save_processed_data", action="store_true", help="save preprocessed data"
    )

    parser.add_argument(
        "--no_validation_set", action="store_true", help="do not use any training data for validation set"
    )

    return parser.parse_args()
