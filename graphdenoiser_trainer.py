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
from autoencoder.model_ae import BaseModel_ae
from autoencoder.model_utils import get_params
from onehot import create_node_mask, update_adj_matrix, one_hot_encode_adj, convert_to_onehot, validate_smiles
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
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles


def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)
    train_loader, test_loader = load_data(config)
    params = get_params()
    model = BaseModel_ae(
        params,
        config.graphdenoiser.epoch_every,
        config.data,
        hidden_size=config.graphdenoiser.hidden_size,
        num_heads=config.graphdenoiser.num_heads,
        depth=config.graphdenoiser.depth,
        using_lincs=False,

    )
    now = str(datetime.now()).replace(" ", "_").replace(":", "_")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(save_dir=f"lightning_logs/{now}", name=f"logs_{now}")
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    timer = Timer(duration="00:12:00:00")  # 12 hours (for training one epoch to check training speed)
    # zutils.create_folders('diff-model')

    # now = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirpath = f"./lightning_logs/{now}"

    # Ensure the directory exists
    os.makedirs(dirpath, exist_ok=True)

    # ModelCheckpoint Callback
    checkpoint_callback = ModelCheckpoint(

        dirpath=dirpath,
        filename="{epoch:02d}-{val_loss:.2f}"

    )
    callbacks = (
        [checkpoint_callback, lr_monitor, timer]
    )

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices= 1 if use_gpu else 1,
        max_epochs=config.graphdenoiser.epoch_all,
        callbacks=callbacks,
        logger=tensorboard_logger,
        gradient_clip_val=1,
        # detect_anomaly=True,
        # track_grad_norm=int(sys.argv[3]), # set to 2 for l2 norm
    )  # overfit_batches=1)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check_point = torch.load(
    #     '/ckpoint/qm9/vae_epoch=19-val_loss=2.82.ckpt',
    # )
    # model.load_state_dict(check_point['state_dict'])
    # model.to(device)
    # model.eval()

    # for batch in train_loader:
    #
    #     x, adj=batch[0].to(device),batch[1].to(device)
    #     node_mask = create_node_mask(x)
    #
    #     adj_update = update_adj_matrix(adj, node_mask)
    #     one_hot_adj = one_hot_encode_adj(adj_update, 4)
    #
    #     # Obtain graph level representation of the partial graph
    #
    #     input_molecule_representations = model.encoder(x, one_hot_adj, node_mask).z
    #     # Apply latent sampling strategy
    #     mu, log_var, latent_representation = model.sample_from_latent_repr(
    #         input_molecule_representations
    #     )
    #
    #     decoder_out = model.decoder(latent_representation, node_mask)
    #     final_x=convert_to_onehot(decoder_out.X)
    #     final_e=decoder_out.E.permute(0, 3, 1, 2)
    #
    #     final_x = torch.concat([final_x, 1 - final_x.sum(dim=-1, keepdim=True)], dim=-1)*node_mask.unsqueeze(-1)
    #
    #     gen_mols, num_mols_wo_correction = gen_mol(final_x, final_e,node_mask, 'QM9')
    #
    #     num_mols = len(gen_mols)
    #
    #     gen_smiles = mols_to_smiles(gen_mols)
    #     gen_smiles = [smi for smi in gen_smiles if len(smi)]
    #
    #
    #
    #
    #
    #     train_smiles, test_smiles = load_smiles('QM9')
    #     train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
    #     scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device, n_jobs=8, test=test_smiles,
    #                              train=train_smiles)
    #     print(scores)






if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
