from trainer  import trainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='eub3', help="name model")
    parser.add_argument("--ds", type=str,  choices=['breast', 'vocalfolds','busi','breast_albumentation'],default="breast")
    args = parser.parse_args()
    trainer(args)