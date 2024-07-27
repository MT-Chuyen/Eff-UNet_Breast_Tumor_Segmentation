from trainer  import trainer
import argparse

if __name__ == '__main__':
 

    parser = argparse.ArgumentParser()
 
    parser.add_argument("--model_name", type=str,  choices=['eub7', 'eub7without2'],default="eub7", help="Model name")
 
    args = parser.parse_args()
    trainer(args)