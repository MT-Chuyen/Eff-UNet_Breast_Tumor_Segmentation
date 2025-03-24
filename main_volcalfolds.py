from trainer  import trainer
from train_vocalfolds import trainer_vocalfolds
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  choices=['eub7', 'eub7without2', 'eub7without3', 'eub7without4', 'eub7without6','unet'],default="eub7", help="Model name")
    parser.add_argument("--ds", type=str,  choices=['breast', 'vocalfolds','busi','breast_albumentation'],default="vocalfolds")
    args = parser.parse_args()
    trainer_vocalfolds(args)