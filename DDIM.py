import sys 
sys.path.append('ldm/')
sys.path.append('../')

import torch
# from datasets import LincsDataset
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
# from model_utils import get_params
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class MolSampler(DDIMSampler):
    def __init__(self, model,vae,gn, schedule="linear", **kwargs):
        super().__init__(model, vae,gn,schedule)
        
    @torch.no_grad()
    def sample(self,
                node_mask,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1,
               noise_dropout=1.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
              ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        # C, H, W = shape    
        # size = (batch_size, C, H, W)
        C, H = shape    # our latent repr is 1d
        size = (batch_size, C, H)

        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,node_mask=node_mask,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,

                                                    )
        return samples, intermediates
    
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, node_mask,
                      repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        # 直接预测 x0（不是预测噪声）
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            pred_x0 = self.model.apply_model(x, t, c, node_mask)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            pred_x0_uncond, pred_x0 = self.model.apply_model(x_in, t_in, c_in, node_mask).chunk(2)
            pred_x0 = pred_x0_uncond + unconditional_guidance_scale * (pred_x0 - pred_x0_uncond)

        K = 8
        guidance_steps = set(map(int, torch.linspace(0, T - 1, K, dtype=torch.long).tolist()))
        idx = int(index) if isinstance(index, torch.Tensor) else index

        if idx in guidance_steps:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)

                decoder_out = self.vae.decoder(x, node_mask)
                denoised_z = self.gn.encoder(decoder_out.X, decoder_out.E, node_mask).z
                decoder_out = self.gn.decoder(denoised_z, node_mask)
                z = self.vae.encoder(decoder_out.X, decoder_out.E, node_mask).z

                mu, log_var, latent_representation = self.vae.sample_from_latent_repr(z)
                latent_representation = latent_representation * node_mask.unsqueeze(-1)

                pred = latent_representation.norm(dim=-1).sum(dim=-1).mean()
                pred = -pred

                grad = torch.autograd.grad(pred, x, retain_graph=True)[0]
                guidance_scale = 0.5
                pred_x0 = pred_x0 - guidance_scale * grad

        # ===============================================

        # 准备DDIM相关参数
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        a_t = torch.full((b, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # 使用pred_x0反推e_t（即用来计算dir_xt）
        e_t = (x - a_t.sqrt() * pred_x0) / sqrt_one_minus_at

        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

        