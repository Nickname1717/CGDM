import torch
from sde import VPSDE, VESDE, subVPSDE
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise


def get_score_fn(sde, model, train=True, continuous=True):

  if not train:
    model.eval()
  model_fn = model

  if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
    def score_fn(x, adj, flags, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        score = model_fn(x, flags)
        std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
      else:
        raise NotImplementedError(f"Discrete not supported")
      score = -score / std[:, None, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, flags, t):
      if continuous:
        score = model_fn(x, t, flags)
      else:  
        raise NotImplementedError(f"Discrete not supported")
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

  return score_fn


def get_sde_loss_fn(sde_x, train=True, reduce_mean=False, continuous=True,
                    likelihood_weighting=False, eps=1e-5):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model_x, x,node_mask):

    score_fn_x = get_score_fn(sde_x, model_x, train=train, continuous=continuous)
    # score_fn_adj = get_score_fn(sde_adj, model_adj, train=train, continuous=continuous)

    t = torch.rand(x.shape[0], device=x.device) * (sde_x.T - eps) + eps
    flags = node_mask.int()

    z_x = gen_noise(x, flags, sym=False)
    mean_x, std_x = sde_x.marginal_prob(x, t)
    perturbed_x = mean_x + std_x[:, None, None] * z_x
    perturbed_x = mask_x(perturbed_x, flags)



    score_x = score_fn_x(perturbed_x, flags, t)


    if not likelihood_weighting:
      losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
      losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)


    else:
      g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
      losses_x = torch.square(score_x + z_x / std_x[:, None, None])
      losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x



    return torch.mean(losses_x)

  return loss_fn