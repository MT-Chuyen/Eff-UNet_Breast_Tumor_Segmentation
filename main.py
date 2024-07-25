from trainer.trainer import trainer

import argparse
import numpy as np
import torch
import random
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
 
    parser.add_argument('-m', '--model', choices=['WITHOUT_2', 'WITHOUT_3','WITHOUT_4','WITHOUT_6'], default='WITHOUT_2')
 
    args = parser.parse_args()
    trainer(args)