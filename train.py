import yaml
import argparse
from utils.trainer import Trainer
from utils.func import random_seed
import os
# Set the CUDA device to 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='configs/DSEC_Semantic.yaml',
                        help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Random seed
    random_seed(cfg['SEED_NUM'])

    # Initialize the trainer
    trainer = Trainer(cfg)
    trainer.train()
