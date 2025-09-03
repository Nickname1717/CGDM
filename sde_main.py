import pickle

import torch
import argparse
import time
import warnings
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer
from mini_moses.metrics.metrics import get_all_metrics
import zutils
from DDIM import MolSampler
from autoencoder.model_ae import BaseModel_ae
from autoencoder.model_utils import get_params
from evaluation.stats import eval_graph_list
from ldm_sde_trainer import Trainer_ldm
from metric.distributions import DistributionNodes
from moler_ldm1 import LatentDiffusion1
from onehot import convert_to_onehot
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler, Sampler_mol
import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from pytorch_lightning import Trainer
from autoencoder.vae_model import BaseModel
from utils import logger
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
    load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx


def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config(args.config, args.seed)
    train_loader, test_loader = load_data(config)
    ldm_params = config.ldmModel.params
    params = config.vae.params
    with open(f'data/{config.data.data.lower()}_test_nx.pkl', 'rb') as f:
        test_graph_list = pickle.load(f)

    model = BaseModel(
        params,
        config.data,
        hidden_size=config.vae.hidden_size,
        num_heads=config.vae.num_heads,
        depth=config.vae.depth,
        using_lincs=False,

    )
    check_point = torch.load(
        '/media/disk1/Projects/zjc/GDSS/lightning_logs/2024-07-28_19_57_07.335999/vae_epoch=09-val_loss=2.81.ckpt',
    )
    model.load_state_dict(check_point['state_dict'])
    model.to(device)
    model.eval()
    trainer=Trainer_ldm(config,model)
    ckpt = trainer.train(ts)

    print(1)














if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
