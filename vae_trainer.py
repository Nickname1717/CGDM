import math

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
from autoencoder.model_utils import get_params
from evaluation.stats import eval_graph_list
from onehot import create_node_mask, update_adj_matrix, one_hot_encode_adj, convert_to_onehot, validate_smiles
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler, Sampler_mol
import os
import time
from tqdm import tqdm, trange
import numpy as np
import csv
from pytorch_lightning import Trainer
from autoencoder.vae_model import BaseModel
from utils.graph_utils import adjs_to_graphs, quantize_mol
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
    load_ema, load_loss_fn, load_batch, load_eval_settings
from utils.logger import Logger, set_log, start_log, train_log
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles




def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)
    train_loader, test_loader = load_data(config)
    params = config.vae.params
    model = BaseModel(
        params,
        config.data,
        hidden_size=config.vae.hidden_size,
        num_heads=config.vae.num_heads,
        depth=config.vae.depth,
        using_lincs=False,

    )
    now = str(datetime.now()).replace(" ", "_").replace(":", "_")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(save_dir=f"lightning_logs/{now}", name=f"logs_{now}")
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    timer = Timer(duration="00:12:00:00")  # 12 hours (for training one epoch to check training speed)



    dirpath = f"./lightning_logs/{now}"

    # Ensure the directory exists
    os.makedirs(dirpath, exist_ok=True)

    # ModelCheckpoint Callback
    checkpoint_callback = ModelCheckpoint(

        dirpath=dirpath,
        filename="vae_{epoch:02d}-{val_loss:.2f}",

        every_n_epochs=1,
    )
    callbacks = (
        [checkpoint_callback, lr_monitor, timer]
    )

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices= 1 if use_gpu else 1,
        max_epochs=config.vae.epoch,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check_point = torch.load(
    #     '/media/disk1/Projects/zjc/GDSS/lightning_logs/2024-09-12_20_52_21.374898/vae_epoch=39-val_loss=3.62.ckpt',
    # )
    # model.load_state_dict(check_point['state_dict'])
    model.to(device)
    model.eval()
    latent_representations = []
    if config.data.data in ['QM9', 'ZINC250k']:

        output_file = "latent_representation_smiles_masked.csv"

        # Check if the file already exists, to avoid writing the header multiple times
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i, batch in enumerate(train_loader):
                if(i==1):
                    break;

                x, adj = batch[0].to(device), batch[1].to(device)
                node_mask = create_node_mask(x)
                x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
                e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
                e_mask2 = x_mask.unsqueeze(1)
                adj_update = update_adj_matrix(adj, node_mask)
                one_hot_adj = one_hot_encode_adj(adj_update, config.data.edge_feat)

                # Obtain graph level representation of the partial graph

                input_molecule_representations = model.encoder(x, one_hot_adj, node_mask).z
                # Apply latent sampling strategy
                mu, log_var, latent_representation = model.sample_from_latent_repr(
                    input_molecule_representations
                )



                decoder_out = model.decoder(latent_representation, node_mask)

                final_x = convert_to_onehot(x)
                final_e = one_hot_adj.permute(0, 3, 1, 2)
                # x_pre=final_x*x_mask
                adj_pre=torch.argmax(one_hot_adj,dim=-1)
                adj_pre[(e_mask1 * e_mask2).squeeze(-1) == 0] = 0

                # accuracy = compute_reconstruction_accuracy(x, x_pre, adj_update, adj_pre, node_mask)
                # print(f"Reconstruction Accuracy: {accuracy:.4f}")


                final_x = torch.concat([final_x, 1 - final_x.sum(dim=-1, keepdim=True)], dim=-1) * x_mask


                # final_x = convert_to_onehot(decoder_out.X)
                # final_e = decoder_out.E.permute(0, 3, 1, 2)
                # # x_pre=final_x*x_mask
                # adj_pre=torch.argmax(decoder_out.E,dim=-1)
                # adj_pre[(e_mask1 * e_mask2).squeeze(-1) == 0] = 0
                #
                # # accuracy = compute_reconstruction_accuracy(x, x_pre, adj_update, adj_pre, node_mask)
                # # print(f"Reconstruction Accuracy: {accuracy:.4f}")
                #
                #
                # final_x = torch.concat([final_x, 1 - final_x.sum(dim=-1, keepdim=True)], dim=-1) * x_mask

                gen_mols, num_mols_wo_correction = gen_mol(final_x, final_e, node_mask, config.data.data)

                num_mols = len(gen_mols)

                gen_smiles = mols_to_smiles(gen_mols)
                gen_smiles = [smi for smi in gen_smiles if len(smi)]

                latent_representation_masked = latent_representation * node_mask.unsqueeze(-1)

                # Convert masked latent representations to numpy
                latent_representation_masked_np = latent_representation_masked.cpu().detach().numpy()

                # Write each SMILES and its corresponding masked latent vector
                if i == 0 and not file_exists:
                    writer.writerow(
                        ['SMILES'] + [f'latent_{i}' for i in range(latent_representation_masked_np.shape[2])])

                    # Write each SMILES and its corresponding masked latent vector
                for smi, latent_vec in zip(gen_smiles, latent_representation_masked_np):
                    writer.writerow([smi] + latent_vec.tolist())

                # Specify the output file name


                # train_smiles, test_smiles = load_smiles(config.data.data)
                # train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
                # scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device, n_jobs=8, test=test_smiles,
                #                          train=train_smiles)
                #
                # print(scores)
                # print(num_mols_wo_correction / num_mols)
                # break;
            # latent_representations = torch.cat(latent_representations, dim=0)
            # torch.save(latent_representations, 'vae_out.pt')
    else:
        train_graph_list, test_graph_list = load_data(config, get_graph_list=True)
        # num_sampling_rounds = math.ceil(len(test_graph_list) / config.data.batch_size)
        # gen_graph_list = []
        for i, batch in enumerate(train_loader):
            # print(i)
            # if i>=6:
            #    break;
            x, adj = batch[0].to(device), batch[1].to(device)
            node_mask = create_node_mask(x)
            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)
            adj_update = update_adj_matrix(adj, node_mask)
            one_hot_adj = one_hot_encode_adj(adj_update, config.data.edge_feat)

            # Obtain graph level representation of the partial graph

            input_molecule_representations = model.encoder(x, one_hot_adj, node_mask).z
            # Apply latent sampling strategy
            mu, log_var, latent_representation = model.sample_from_latent_repr(
                input_molecule_representations
            )
            # latent_representations.append(latent_representation)
            # latent_representations = torch.cat(latent_representations, dim=0)
            # torch.save(latent_representations,'vae_out.pt')
            decoder_out = model.decoder(latent_representation, node_mask)
            final_x = convert_to_onehot(decoder_out.X)*x_mask
            # self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
            adj_pre=torch.argmax(decoder_out.E,dim=-1)
            adj_pre[(e_mask1 * e_mask2).squeeze(-1) == 0] = 0
            gen_graph_list = []
            gen_graph_list.extend(adjs_to_graphs(adj_pre, True))


            gen_graph_list = gen_graph_list[:len(test_graph_list)]

            # -------- Evaluation --------
            methods, kernels = load_eval_settings(config.data.data)
            result_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=methods, kernels=kernels)

            torch.cuda.empty_cache()









if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
