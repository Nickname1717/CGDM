import math
import pickle
import csv
import os
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
from utils.graph_utils import adjs_to_graphs, init_flags, quantize_mol
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
    load_ema, load_loss_fn, load_batch, load_eval_settings
from utils.logger import Logger, set_log, start_log, train_log
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from utils.plot import plot_graphs_list, save_graph_list


def main(work_type_args):

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config(args.config, args.seed)
    train_loader, test_loader = load_data(config)
    ldm_params = config.ldmModel.params
    params = config.vae.params
    if config.data.data in ['QM9', 'ZINC250k']:
        with open(f'data/{config.data.data.lower()}_test_nx.pkl', 'rb') as f:
            test_graph_list = pickle.load(f)
    else:
        train_graph_list, test_graph_list = load_data(config, get_graph_list=True)


    model = BaseModel(
        params,
        config.data,
        hidden_size=config.vae.hidden_size,
        num_heads=config.vae.num_heads,
        depth=config.vae.depth,
        using_lincs=False,

    )
    check_point = torch.load(
        '/media/disk1/Projects/zjc/GDSS/lightning_logs/2024-12-18_14_03_13.959685/vae_epoch=19-val_loss=5.35.ckpt',
    )
    model.load_state_dict(check_point['state_dict'])
    model.to(device)
    # model.eval()
    #
    graph_model = BaseModel_ae(
        params,
        config.graphdenoiser.epoch_every,
        config.data,
        hidden_size=config.graphdenoiser.hidden_size,
        num_heads=config.graphdenoiser.num_heads,
        depth=config.graphdenoiser.depth,
        using_lincs=False,

    )
    check_point1 = torch.load(
        '/media/disk1/Projects/zjc/GDSS/lightning_logs/2025-07-21_22_41_52.579001/epoch=14-val_loss=3.20.ckpt',
    )
    graph_model.load_state_dict(check_point1['state_dict'])
    graph_model.to(device)
    graph_model.eval()

    lr = config.ldmModel.base_learning_rate

    now = str(datetime.now()).replace(" ", "_").replace(":", "_")

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # tensorboard_logger = TensorBoardLogger(save_dir=f"lightning_logs/{now}", name=f"logs_{now}")
    early_stopping = EarlyStopping(monitor=ldm_params.monitor, patience=3)
    timer = Timer(duration="00:12:00:00")  # 12 hours (for training one epoch to check training speed)
    use_gpu = torch.cuda.is_available()
    checkpoint_callback = ModelCheckpoint(
        # save_top_k=1,
        # monitor="val/loss",
        dirpath=f"lightning_logs/{now}",
        # mode="min",
        every_n_train_steps=1,
        filename='ldm_epoch={epoch:02d}-step={global_step}-val_loss={val/loss:.2f}',
        auto_insert_metric_name=False,
    )
    callbacks = (
        [checkpoint_callback, lr_monitor,timer,early_stopping]
        # if model_architecture == "vae"
        # else [checkpoint_callback, lr_monitor]
    )
    ldm_model = LatentDiffusion1(model,config.data, unet_config=config.ldmModel.unet_config, **ldm_params)
    ldm_model.learning_rate = lr
    trainer = Trainer(accelerator='gpu' if use_gpu else 'cpu',
                      devices= 1 if use_gpu else 1,
                      max_epochs=config.ldmModel.epoch,
                      #   num_sanity_val_steps=0,    # the CUDA capability is insufficient to train the whole batch, we drop some graphs in each batch, but need to set num_sanity_val_steps=0 to avoid the validation step to run (with the whole batch)

                      callbacks=callbacks,
                      # logger=tensorboard_logger,
                      gradient_clip_val=1)
    # trainer.fit(ldm_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    #
    ckpt_path = '/media/disk1/Projects/zjc/GDSS/lightning_logs/2025-07-30_17_02_48.662844/ldm_epoch=73-step=1775.0-val_loss=7.40.ckpt'
    checkpoint = torch.load(ckpt_path)
    ldm_model.load_state_dict(checkpoint['state_dict'])

    ldm_model.to(device)
    ldm_model.eval()
    sampler = MolSampler(ldm_model, model, graph_model)
    if config.data.data in ['QM9', 'ZINC250k']:
        n_samples = 1000
        ddim_steps = 1000
        ddim_eta = 1
        size = (config.data.max_node_num, config.vae.hidden_size_latent)

        nodes_dist=DistributionNodes(torch.tensor(config.data.n_nodes))

        n_nodes = nodes_dist.sample_n(n_samples, 'cpu')

        n_nodes_max = config.data.max_node_num

        # Build the masks
        arange = torch.arange(n_nodes_max, device='cpu').unsqueeze(0).expand(n_samples, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        node_mask = node_mask.to(device)
        samples, _ = sampler.sample(

            node_mask=node_mask,
            S=ddim_steps,
            batch_size=n_samples,  # not batch size
            shape=size,
            ddim_eta=ddim_eta
        )
        torch.save(samples, 'samples.pt')
        decoder_out = ldm_model.vae.decoder(samples, node_mask)
        # print(pos)
        # with torch.no_grad():
        #
        #     decoder_out = model.decoder(samples, node_mask)
        #     denoised_z = graph_model.encoder(decoder_out.X, decoder_out.E, node_mask).z
        #
        #     for i in range(2):
        #         decoder_out=graph_model.decoder(denoised_z,node_mask)
        #         denoised_z=graph_model.encoder(decoder_out.X, decoder_out.E, node_mask).z
        #
        #     torch.save(denoised_z,'denoised_z.pt')
        #
        #     decoder_out = graph_model.decoder(denoised_z, node_mask)

        final_x = convert_to_onehot(decoder_out.X)
        final_e = decoder_out.E.permute(0, 3, 1, 2)

        final_x = torch.concat([final_x, 1 - final_x.sum(dim=-1, keepdim=True)], dim=-1) * node_mask.unsqueeze(-1)


        gen_mols, num_mols_wo_correction = gen_mol(final_x, final_e, node_mask, config.data.data)

        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)


        # output_file = "samples_and_smiles.csv"
        #
        # # Convert samples to numpy for saving
        # samples_np = samples.cpu().detach().numpy()
        #
        # with open(output_file, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #
        #     # Write the header
        #     writer.writerow(['SMILES'] + [f'sample_{i}' for i in range(samples_np.shape[2])])
        #
        #     # Write each SMILES and corresponding sample vector
        #     for smi, sample_vec in zip(gen_smiles, samples_np):
        #         writer.writerow([smi] + sample_vec.tolist())
        #
        # print(f"Samples and SMILES saved to {output_file}")

        with open('gensmiles.txt', 'w') as f:
            for smile in gen_smiles:
                f.write(smile + '\n')

        print("SMILES 已成功保存为 gensmiles.txt")
        train_smiles, test_smiles = load_smiles(config.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device, n_jobs=8, test=test_smiles,
                                 train=train_smiles)

        scores_nspdk = eval_graph_list(test_graph_list, mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
        print(scores)
        wo_correction=num_mols_wo_correction / num_mols
        print(wo_correction)
        print(scores_nspdk)
        # logger.log(f'Number of molecules: {num_mols}')
        # logger.log(f'validity w/o c: {num_mols_wo_correction / num_mols}')
        # for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']:
        #     logger.log(f'{metric}: {scores[metric]}')
        # logger.log(f'NSPDK MMD: {scores_nspdk}')
        # logger.log('=' * 100)
    else:
        num_sampling_rounds = math.ceil(len(test_graph_list) / config.data.batch_size)
        gen_graph_list = []
        ddim_steps = 1000
        ddim_eta = 1
        size = (config.data.max_node_num, config.vae.hidden_size_latent)

        for r in range(num_sampling_rounds):
            node_mask = init_flags(train_graph_list, config).bool().to(device)
            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)
            n_samples = len(node_mask)
            samples, _ = sampler.sample(

                node_mask=node_mask,
                S=ddim_steps,
                batch_size=n_samples,  # not batch size
                shape=size,
                ddim_eta=ddim_eta
            )
            # torch.save(samples, 'samples.pt')
            # decoder_out = model.decoder(samples, node_mask)
            with torch.no_grad():

                decoder_out = model.decoder(samples, node_mask)
                denoised_z = graph_model.encoder(decoder_out.X, decoder_out.E, node_mask).z

                for i in range(0):
                    decoder_out=graph_model.decoder(denoised_z,node_mask)
                    denoised_z=graph_model.encoder(decoder_out.X, decoder_out.E, node_mask).z

                # torch.save(denoised_z,'denoised_z.pt')

                decoder_out = graph_model.decoder(denoised_z, node_mask)
            adj_pre = torch.argmax(decoder_out.E, dim=-1)
            adj_pre[(e_mask1 * e_mask2).squeeze(-1) == 0] = 0
            gen_graph_list.extend(adjs_to_graphs(adj_pre, True))
        gen_graph_list = gen_graph_list[:len(test_graph_list)]

        # -------- Evaluation --------
        methods, kernels = load_eval_settings(config.data.data)
        result_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=methods, kernels=kernels)
        print(result_dict)
        save_dir = save_graph_list('samples', 1, gen_graph_list)
        with open(save_dir, 'rb') as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(graphs=sample_graph_list, title=f'1', max_num=16,
                         save_dir=save_dir)










if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
