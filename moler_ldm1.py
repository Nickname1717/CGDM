import sys
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
import zutils
from dit.dit import DiT
from dit.dittest import DiT1
from dit.udit_models import U_DiT, U_DiT_S, U_DiT_custom, U_DiT_zink250
from TUD.transformer import TUD
from onehot import create_node_mask, update_adj_matrix, one_hot_encode_adj

sys.path.append('ldm/')
import torch
from ldm.models.diffusion.ddpm import DDPM, DiffusionWrapper, disabled_train
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, \
    get_obj_from_str
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities import rank_zero_only
import numpy as np


class LatentDiffusion1(DDPM):
    def __init__(self,
                 vae,
                 data_info,
                 unet_config,
                 num_timesteps_cond=None,
                 cond_stage_key="gene_expressions",
                 cond_stage_trainable=False,  # we can either use pretrained MLP or train a new one
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,  # by default, concat mode is used
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        # self.log("drop_prob", datasets._gen_step_drop_probability)    # can't call here since trainer is not initiated yet

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        # assert self.num_timesteps_cond <= kwargs['timesteps']
        self.data_info=data_info

        # to init the first stage model
        self.latent_dim = 16
        # self.datasets = datasets
        self.batch_size = 512
        # self.model_architecture = first_stage_config['model_type']
        # self.first_stage_params = first_stage_params
        # self.first_stage_ckpt = first_stage_ckpt

        # for backwards compatibility after implementation of DiffusionWrapper
        # if conditioning_key is None:
        #     conditioning_key = 'concat' if concat_mode else 'crossattn'
        # self.model = DiffusionWrapper(unet_config, conditioning_key)
        ckpt_path = kwargs.pop("ckpt_path", None)  # this is for the diff model ckpt, not vaes
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, unet_config=unet_config, *args,
                         **kwargs)  # DiffusionWrapper is called here, and the diffusion model (target in the unet config, aka the unet model) is initiated and called in the DiffusionWrapper
        # self.save_hyperparameters(ignore=['datasets'])

        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        # self.instantiate_first_stage(first_stage_config, first_stage_params, dataset, first_stage_ckpt)    # first stage model is initiated here
        # self.instantiate_cond_stage(cond_stage_config)    # let's remove this and directly input the gene expr to unet
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.learning_rate = 5.0e-03
        self.vae = vae
        self.train_loss = zutils.TrainLossDiscrete([10])

        self.model = TUD(input_size=16)

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        # this function should be called at the beginning of trainer.fit_loop, not sure if it will be called in trainer.fit
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            # x = super().get_input(batch, self.first_stage_key)    # this is not necessary as the whole batch is always passed together
            x = batch
            x = x.to(self.device)
            encoder_posterior, partial_reprs, node_reprs = self.encode_first_stage(x)
            self.partial_reprs = partial_reprs
            self.node_reprs = node_reprs

            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    # def init_first_stage_model(self, config, ckpt, **kwargs):
    #     # vae kwargs: params, datasets, using_lincs, include_predict_gene_exp_mlp = False, num_train_batches=1, batch_size=1, use_clamp_log_var = False
    #     if not "target" in config:
    #         if config == '__is_first_stage__':
    #             return None
    #         elif config == "__is_unconditional__":
    #             return None
    #         raise KeyError("Expected key `target` to instantiate.")
    #     model = get_obj_from_str(config["target"])
    #     model = model.load_from_checkpoint(ckpt, **kwargs)
    #
    #     return model

    #     def instantiate_first_stage(self, config, params, dataset, ckpt):
    # #         model = instantiate_from_config(config)    # TODO: rewrite with our model init function
    #
    #         if config['model_type'] == 'vae':
    #             model = self.init_first_stage_model(config, ckpt, params=params, dataset=dataset, using_lincs=config['using_lincs'])
    #         elif config['model_type'] == 'aae':
    #             model = self.init_first_stage_model(
    #                 config,
    #                 ckpt,
    #                 params=params,
    #                 dataset=dataset,
    #                 using_lincs=config['using_lincs'],
    #                 using_wasserstein_loss=False,
    #                 using_gp=False,
    #             )
    #         elif config['model_type'] == 'wae':
    #             model = self.init_first_stage_model(
    #                 config,
    #                 ckpt,
    #                 params=params,
    #                 dataset=dataset,
    #                 using_lincs=config['using_lincs'],
    #                 using_wasserstein_loss=True,
    #                 using_gp=True,
    #             )
    #         else:
    #             raise NotImplementedError('first stage model type is not supported')
    #
    #         self.first_stage_model = model.eval()
    #         self.first_stage_model.train = disabled_train    # overwrite model's train function with the empty disabled_train
    #         for param in self.first_stage_model.parameters():
    #             param.requires_grad = False

    '''
    def instantiate_cond_stage(self, config):
        # TODO: adapt this function
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
        '''

    #     def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
    #         # we may not need the denoising row?
    #         denoise_row = []
    #         for zd in tqdm(samples, desc=desc):
    #             denoise_row.append(self.decode_first_stage(zd.to(self.device),
    #                                                             force_not_quantize=force_no_decoder_quantization))
    #         n_imgs_per_row = len(denoise_row)
    #         denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
    #         denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
    #         denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    #         denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
    #         return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        # we don't have the diagonal gaussian dist
        # if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        #     z = encoder_posterior.sample()
        mu = self.vae.fc_mu(encoder_posterior)
        log_var = self.vae.fc_logvar(encoder_posterior)

        z = self.vae.reparameterize(mu, log_var)

        return self.scale_factor * z

    '''
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            # TODO: if our cond model is a GenericMLP, set self.cond_stage_forward as the forward function of self.cond_stage_model
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
    '''

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        # TODO: rewrite this function to get the mol input and the gene expr input
        # TODO: this function is called in shared_step (used in training/val_step for lightening model). adapt them as well
        '''
        this function is to get inputs for the diff model.
        It should only return the latent repr to get diffused and the cond vector to control the diffusion

        output of this function:
            z: latent repr of the first stage model
            partial_repr?? no need to have this
            c: output of the cond stage model (or it can be the gene expr directly??)
        '''

        # x = super().get_input(batch, k)

        x, adj = batch[0], batch[1]

        node_mask = create_node_mask(x)

        adj_update = update_adj_matrix(adj, node_mask)
        one_hot_adj = one_hot_encode_adj(adj_update, self.data_info.edge_feat)

        # Obtain graph level representation of the partial graph

        input_molecule_representations = self.vae.encoder(x, one_hot_adj, node_mask).z
        # Apply latent sampling strategy
        mu, log_var, latent_representation = self.vae.sample_from_latent_repr(
            input_molecule_representations
        )
        latent_representation=latent_representation*node_mask.unsqueeze(-1)
        return latent_representation,node_mask,x,one_hot_adj

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        z = 1. / self.scale_factor * z
        (
            first_node_type_logits,
            node_type_logits,
            edge_candidate_logits,
            edge_type_logits,
            attachment_point_selection_logits,
        ) = self.first_stage_model.decoder(
            input_molecule_representations=z,
            graph_representations=self.partial_reprs,
            graphs_requiring_node_choices=self.batch.correct_node_type_choices_batch.unique(),
            # edge selection
            node_representations=self.node_reprs,
            num_graphs_in_batch=len(self.batch.ptr) - 1,
            focus_node_idx_in_batch=self.batch.focus_node,
            node_to_graph_map=self.batch.batch,
            candidate_edge_targets=self.batch.valid_edge_choices[:, 1].long(),
            candidate_edge_features=self.batch.edge_features,
            # attachment selection
            candidate_attachment_points=self.batch.valid_attachment_point_choices.long(),
        )
        return [first_node_type_logits, node_type_logits, edge_candidate_logits, edge_type_logits,
                attachment_point_selection_logits]

    @torch.no_grad()
    def encode_first_stage(self, batch):
        # other repr from partial encoder should also be done and passed to self, so that decoder can access to them
        input_molecule_representations = self.first_stage_model.full_graph_encoder(
            original_graph_node_categorical_features=batch.original_graph_node_categorical_features,
            node_features=batch.original_graph_x.float(),
            edge_index=batch.original_graph_edge_index,
            edge_features=batch.original_graph_edge_features,  # can be edge_type or edge_attr
            batch_index=batch.original_graph_x_batch,
        )
        partial_graph_representations, node_representations = self.first_stage_model.partial_graph_encoder(
            partial_graph_node_categorical_features=batch.partial_node_categorical_features,
            node_features=batch.x,
            edge_index=batch.edge_index.long(),
            edge_features=batch.partial_graph_edge_features,
            graph_to_focus_node_map=batch.focus_node,
            candidate_attachment_points=batch.valid_attachment_point_choices,
            batch_index=batch.batch,
        )
        return input_molecule_representations, partial_graph_representations, node_representations

    def shared_step(self, batch, batch_id=None, **kwargs):
        # skip a weird batch
        # if batch['dose'].size(0) != 1000:
        #     raise ValueError('channel number is not 1000!')

        # pass the batch data to decoder via self
        self.batch = batch

        # x here is actually latent repr z. it's written as x to be consistent with the DDPM theory.
        z,node_mask,x,one_hot_adj=self.get_input(batch)


        loss = self(z,None,node_mask,x,one_hot_adj)
        return loss

    def forward(self, z,c,node_mask,x,one_hot_adj,*args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()


        # if self.model.conditioning_key is not None:
        #     assert c is not None
        #     if self.cond_stage_trainable:,
        #         c = self.get_learned_conditioning(c)
        #     if self.shorten_cond_schedule:  # TODO: drop this option
        #         tc = self.cond_ids[t].to(self.device)
        #         c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(z, c, t,node_mask,x,one_hot_adj,*args, **kwargs)

    def apply_model(self, x_noisy,t, cond ,node_mask,return_ids=False):
        x_recon = self.model(x_noisy,t,node_mask)
        return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, z, cond, t, node_mask, x, one_hot_adj, noise=None):
        noise_z = default(noise, lambda: torch.randn_like(z)) * node_mask.unsqueeze(-1)
        z_noisy = self.q_sample(x_start=z, t=t, noise=noise_z)

        decoder_out = self.vae.decoder(z, node_mask)
        perturbed_decoder_out = self.vae.decoder(z_noisy, node_mask)

        # VAE 重构损失
        loss_recon = self.train_loss(masked_pred_X=decoder_out.X, masked_pred_E=decoder_out.E,
                                     true_X=x, true_E=one_hot_adj)
        loss_perturbed = self.train_loss(masked_pred_X=perturbed_decoder_out.X, masked_pred_E=perturbed_decoder_out.E,
                                         true_X=x, true_E=one_hot_adj)

        # 合并重构损失（固定权重为1.0）
        recon_loss = loss_recon + loss_perturbed

        model_output = self.apply_model(z_noisy, t, cond, node_mask)

        if self.parameterization == "x0":
            target = z
        elif self.parameterization == "eps":
            target = noise_z
        else:
            raise NotImplementedError()

        loss_simple0 = self.get_loss(model_output, target, mean=False)
        loss_simple = loss_simple0.mean([1, 2]).mean()

        prefix = 'train' if self.training else 'val'
        loss_dict = {f'{prefix}/loss_simple': loss_simple}
        self.logvar=self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        # 主要扩散损失
        loss = self.l_simple_weight * loss.mean()

        # 可选 VLB 项
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss += self.original_elbo_weight * loss_vlb

        # ✅ 加入固定权重的 VAE 重构损失
        loss += 1.0 * recon_loss

        loss_dict.update({
            f'{prefix}/loss_vlb': loss_vlb,
            f'{prefix}/recon_loss': recon_loss,
            f'{prefix}/loss': loss
        })

        return loss, loss_dict

    def compute_cdist_mae(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean absolute error of conformer and conformer_hat.

        Args:
            - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
            - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
            - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.

        Returns:
            torch.Tensor: The mean absolute error of conformer and conformer_hat.
        """
        mae = F.l1_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
        return mae
    def compute_cdist_mse(masked_cdist: torch.Tensor, masked_cdist_hat: torch.Tensor, cdist_mask: torch.Tensor) -> torch.Tensor:
        """Compute root mean squared error of conformer and conformer_hat.

        Args:
            - masked_cdist (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer.
            - masked_cdist_hat (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the pairwise distance matrix of the conformer_hat.
            - cdist_mask (torch.Tensor): A torch tensor of shape (b, l, l), which denotes the mask of the pairwise distance matrix.

        Returns:
            torch.Tensor: The root mean squared error of conformer and conformer_hat.
        """
        mse = F.mse_loss(masked_cdist, masked_cdist_hat, reduction="sum") / cdist_mask.sum()  # exclude padding atoms
        return mse
    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]

        # img is standard normal dist or the last timestep x_T
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))  # add noise to the cond?

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        # we can call this function outside to get the sampled latent reprs
        # x_T can be none, it will be set as randn automatically
        # x0 can also none, it's only applicable when mask is not none. (guess mask is for some inpainting function?)
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_mol(self, batch):
        # adapt log_images to record the generated molecules
        return

    def configure_optimizers(self):
        lr = self.learning_rate
        weight_decay = 1.0e-04

        params = list(self.model.parameters())  # 扩散模型参数

        # 联合优化 VAE 编解码器
        if hasattr(self, "vae"):
            print(f"{self.__class__.__name__}: 联合优化 VAE 参数!")
            # encoder + decoder 参数
            params += list(self.vae.encoder.parameters())
            params += list(self.vae.decoder.parameters())

        # 可选：条件器
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: 也优化 conditioner 参数!")
            params += list(self.cond_stage_model.parameters())

        # 可选：learnable logvar
        if self.learn_logvar:
            print('扩散模型优化 logvar')
            params.append(self.logvar)

        # 优化器
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # 学习率调度器（可选）
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            print("设置 LambdaLR 调度器...")
            scheduler = [{
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
            print("调度器设置完成.")
            return [opt], scheduler

        return opt

