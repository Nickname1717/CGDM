import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from dit.dittest import DiT1
from onehot import create_node_mask, one_hot_encode_adj, update_adj_matrix
from utils import logger
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
    load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log


class Trainer_ldm(object):
    def __init__(self, config,vae):
        super(Trainer_ldm, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config)

        self.params_x, self.params_adj = load_model_params(self.config)
        self.vae=vae
        self.model=DiT1(input_size=16,depth=6)

    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(self.model, self.config.train,
                                                                                self.device)
        # self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(self.params_adj,
        #                                                                               self.config.train,
        #                                                                               self.device)
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        # self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)

        # logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        # logger.log(f'{self.ckpt}', verbose=False)
        # start_log(logger, self.config)
        # train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config)

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc='[Epoch]', position=1, leave=False):

            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            t_start = time.time()

            self.model_x.train()
            # self.model_adj.train()

            for _, train_b in enumerate(self.train_loader):
                self.optimizer_x.zero_grad()
                # self.optimizer_adj.zero_grad()
                x, adj = load_batch(train_b, self.device)
                node_mask = create_node_mask(x)

                adj_update = update_adj_matrix(adj, node_mask)
                one_hot_adj = one_hot_encode_adj(adj_update, 4)

                # Obtain graph level representation of the partial graph

                input_molecule_representations = self.vae.encoder(x, one_hot_adj, node_mask).z
                # Apply latent sampling strategy
                mu, log_var, latent_representation = self.vae.sample_from_latent_repr(
                    input_molecule_representations
                )
                loss_subject = latent_representation

                loss_x = self.loss_fn(self.model_x, loss_subject,node_mask)
                print(loss_x)
                loss_x.backward()


                torch.nn.utils.clip_grad_norm_(self.model_x.parameters(), self.config.train.grad_norm)


                self.optimizer_x.step()


                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())


                self.train_x.append(loss_x.item())


            if self.config.train.lr_schedule:
                self.scheduler_x.step()

            self.model_x.eval()

            for _, test_b in enumerate(self.test_loader):
                x, adj = load_batch(test_b, self.device)
                node_mask = create_node_mask(x)

                adj_update = update_adj_matrix(adj, node_mask)
                one_hot_adj = one_hot_encode_adj(adj_update, 4)

                # Obtain graph level representation of the partial graph

                input_molecule_representations = self.vae.encoder(x, one_hot_adj, node_mask).z
                # Apply latent sampling strategy
                mu, log_var, latent_representation = self.vae.sample_from_latent_repr(
                    input_molecule_representations
                )
                loss_subject = latent_representation

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())


                    loss_x = self.loss_fn(self.model_x, loss_subject,node_mask)
                    print(loss_x)
                    self.test_x.append(loss_x.item())


                    self.ema_x.restore(self.model_x.parameters())


            mean_train_x = np.mean(self.train_x)

            mean_test_x = np.mean(self.test_x)




            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval - 1:
                save_name = f'_{epoch + 1}' if epoch < self.config.train.num_epochs - 1 else ''

                torch.save({
                    'model_config': self.config,
                    'params_x': self.params_x,
                    'x_state_dict': self.model_x.state_dict(),
                    'ema_x': self.ema_x.state_dict(),
                }, f'./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')

            if epoch % self.config.train.print_interval == self.config.train.print_interval - 1:
                tqdm.write(f'[EPOCH {epoch + 1:04d}] '
                           f'test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}')
        print(' ')
        return self.ckpt
