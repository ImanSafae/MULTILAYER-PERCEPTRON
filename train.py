from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(prog="Training", description="Training of a multilayer perceptron model")
    parser.add_argument("--train", help="a .csv preprocessed training dataset", required=True)
    parser.add_argument("--valid", help="a .csv preprocessed validation dataset", required=True)
    args = parser.parse_args()
    train_dataset = args.train
    valid_dataset = args.valid