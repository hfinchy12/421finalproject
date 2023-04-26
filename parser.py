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

    parser.add_argument(
        "--lr", type=float, default=0.00001, help="learning rate for Adam optimizer" 
    )

    parser.add_argument(
        "--batch_size", type=int, default=10, help="Training batch size"
    )

    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of training epochs"
    )

    parser.add_argument(
        "--plot", action="store_true", help="plot average loss and ROC score during training"
    )

    return parser.parse_args()
