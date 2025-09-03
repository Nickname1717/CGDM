import sys
import itertools
from pytorch_lightning import LightningModule

from rdkit.Chem import Draw
from rdkit import Chem
import torch.nn.functional as F
import torch
import numpy as np

from autoencoder.model.gnn import GNNencoder, GNNDecoder
from autoencoder.model.vgae import DiffEncoder, DiffDecoder
from autoencoder.model_utils import GenericMLP

from torchvision import transforms

# from src import utils

import torch.nn.functional as F

# from src.metrics.train_metrics import TrainLossDiscrete

# sys.path.append("./moler_reference")
from molecule_generation.utils.training_utils import get_class_balancing_weights

from molecule_generation.utils.moler_decoding_utils import (
    restrict_to_beam_size_per_mol,
    MoLeRDecoderState,
    MoleculeGenerationAtomChoiceInfo,
    MoleculeGenerationAttachmentPointChoiceInfo,
    MoleculeGenerationEdgeChoiceInfo,
    MoleculeGenerationEdgeCandidateInfo,
)

from onehot import one_hot_encode_adj, create_node_mask, update_adj_matrix
from utils.graph_utils import mask_adjs
from utils.loss import TrainLossDiscrete_new
from zutils import TrainLossDiscrete


class AbstractModel(LightningModule):
    """Common decoding methods for each model (decoding at inference time doesn't change)"""

    def __init__(self):
        super(AbstractModel, self).__init__()

    def _is_atom_type(self, node_type):
        if not self.uses_motifs:
            return True
        else:
            return node_type in self._atom_types

    def _add_atom_or_motif(
        self,
        decoder_state,
        node_type,
        logprob,
        choice_info,
    ):
        # If we are running with motifs, we need to check whether `node_type` is an atom or a motif.
        if self._is_atom_type(node_type):
            return (
                MoLeRDecoderState.new_with_added_atom(
                    decoder_state,
                    node_type,
                    atom_logprob=logprob,
                    atom_choice_info=choice_info,
                ),
                False,
            )
        else:
            return (
                MoLeRDecoderState.new_with_added_motif(
                    decoder_state,
                    node_type,
                    motif_logprob=logprob,
                    atom_choice_info=choice_info,
                ),
                True,
            )

    @property
    def uses_motifs(self):
        return self._uses_motifs

    @property
    def uses_categorical_features(self):
        return self._node_categorical_num_classes is not None

    @property
    def full_graph_encoder(self):
        return self._full_graph_encoder

    @property
    def partial_graph_encoder(self):
        return self._partial_graph_encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def mean_log_var_mlp(self):
        return self._mean_log_var_mlp

    @property
    def latent_dim(self):
        return self._latent_repr_dim




    # def on_validation_epoch_end(self, outputs):
    #     # decoder 50 random molecules using fixed random seed
    #     if self._decode_on_validation_end:
    #         if self.current_epoch < 3:
    #             pass
    #         else:
    #             generator = torch.Generator(
    #                 device=self.full_graph_encoder._dummy_param.device
    #             ).manual_seed(0)
    #             latent_vectors = torch.randn(
    #                 size=(50, 512),
    #                 generator=generator,
    #                 device=self.full_graph_encoder._dummy_param.device,
    #             )
    #             decoder_states = self.decode(latent_representations=latent_vectors)
    #             print(
    #                 [
    #                     Chem.MolToSmiles(decoder_states[i].molecule)
    #                     for i in range(len(decoder_states))
    #                 ]
    #             )
    #             try:
    #                 pil_imgs = [
    #                     Draw.MolToImage(decoder_states[i].molecule)
    #                     for i in range(len(decoder_states))
    #                 ]
    #                 pil_img_tensors = [
    #                     transforms.ToTensor()(pil_img).permute(1, 2, 0)
    #                     for pil_img in pil_imgs
    #                 ]
    #
    #                 for pil_img_tensor in pil_img_tensors:
    #                     self.logger.experiment.add_image(
    #                         "sample_molecules", pil_img_tensor, self.current_epoch
    #                     )
    #             except Exception as e:
    #                 print(e)


class BaseModel(AbstractModel):
    def __init__(self, params,data_info,hidden_size,num_heads,depth,using_lincs , use_clamp_log_var = True):
        """Params is a nested dictionary with the relevant parameters."""
        super(BaseModel, self).__init__()
        # self._init_params(params, dataset)

        if "training_hyperparams" in params:
            self._training_hyperparams = params["training_hyperparams"]
        else:
            self._training_hyperparams = None
        self._params = params

        self._use_clamp_log_var = use_clamp_log_var
        self._use_oclr_scheduler = params["use_oclr_scheduler"]
        self._decode_on_validation_end = params["decode_on_validation_end"]
        self._using_cyclical_anneal = params["using_cyclical_anneal"]
        # Graph encoders
        self.data_info=data_info

        self.encoder = DiffEncoder(max_n_nodes=data_info.max_node_num, Xdim=data_info.max_feat_num,
                                   Edim=data_info.edge_feat, hidden_size=hidden_size, depth=depth, num_heads=num_heads)

        self.decoder = DiffDecoder(max_n_nodes=data_info.max_node_num, Xdim=data_info.max_feat_num,
                                   Edim=data_info.edge_feat, hidden_size=params['latent_repr_dim'], depth=depth, num_heads=num_heads)

        # self.encoder =GNNencoder(max_feat_num=data_info.max_feat_num,depth=3,nhid=256)
        # self.decoder=GNNDecoder(nfeat=16,nhid=256, output_feat_num=4, ain=81,num_layers=3)
        # Replace this with any other latent space mapping techniques eg diffusion
        self._mean_log_var_mlp = GenericMLP(**self._params["mean_log_var_mlp"])
        # active_index = dataset_infos.active_index
        # self.active_index = active_index
        # MLP for regression task on graph properties
        self._include_property_regressors = "graph_properties" in self._params
        # self.train_loss = TrainLossDiscrete_new([1,10])
        self.train_loss=TrainLossDiscrete([10])
        self._batch_size=512
        self.max_n_nodes = data_info.max_node_num
        # params for latent space
        self._latent_sample_strategy = self._params["latent_sample_strategy"]
        self._latent_repr_dim = self._params["latent_repr_size"]
        self._kl_divergence_weight = self._params["kl_divergence_weight"]
        self._kl_divergence_annealing_beta = self._params[
            "kl_divergence_annealing_beta"
        ]
        # If using lincs gene expression
        self._using_lincs = using_lincs




    def _init_params(self, params, dataset):
        """
        Initialise class weights for next node prediction and placefolder for
        motif/node embeddings.
        """

        self._motif_vocabulary = dataset.metadata.get("motif_vocabulary")
        self._uses_motifs = self._motif_vocabulary is not None

        self._node_categorical_num_classes = len(dataset.node_type_index_to_string)

        if self.uses_categorical_features:
            if "categorical_features_embedding_dim" in params:
                self._node_categorical_features_embedding = None

        if self.uses_motifs:
            # Record the set of atom types, which will be a subset of all node types.
            self._atom_types = set(
                dataset._atom_type_featuriser.index_to_atom_type_map.values()
            )

        self._index_to_node_type_map = dataset.node_type_index_to_string
        self._atom_featurisers = dataset._metadata["feature_extractors"]
        self._num_node_types = dataset.num_node_types

    def sample_from_latent_repr(self, latent_repr):
        mean_and_log_var = self.mean_log_var_mlp(latent_repr)
        mean_and_log_var=mean_and_log_var
        # mean_and_log_var = torch.clamp(mean_and_log_var, min=-10, max=10)
        # perturb latent repr
        mu = mean_and_log_var[:,:, : self.latent_dim]  # Shape: [V, MD]
        log_var = mean_and_log_var[:,:, self.latent_dim :]  # Shape: [V, MD]

        # result_representations: shape [num_partial_graphs, latent_repr_dim]
        z = self.reparametrize(mu, log_var)
        # p, q, z = self.reparametrize(mu, log_var)

        return mu, log_var, z
        # return p, q, z


    
    def reparametrize(self, mu, log_var):
        """Samples a different noise vector for each partial graph.
        TODO: look into the other sampling strategies."""

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # q = torch.distributions.Normal(mu, std)
        # z = q.rsample()
        # return p, q, z

    def forward(self, batch):
        moler_output = self._run_step(batch)
        return (
            moler_output.first_node_type_logits,
            moler_output.node_type_logits,
            moler_output.edge_candidate_logits,
            moler_output.edge_type_logits,
            moler_output.attachment_point_selection_logits,
        )

    def _run_step(self, batch):
        # Obtain graph level representation of original molecular graph

        x, adj=batch[0],batch[1]


        node_mask =create_node_mask(x)



        adj_update = update_adj_matrix(adj,node_mask)
        one_hot_adj = one_hot_encode_adj(adj_update, self.data_info.edge_feat)









        # Obtain graph level representation of the partial graph

        input_molecule_representations=self.encoder(x,one_hot_adj,node_mask).z
        # Apply latent sampling strategy
        mu, log_var, latent_representation = self.sample_from_latent_repr(
            input_molecule_representations
        )

        eps = torch.randn_like(latent_representation)
        perturbed_latent_representation = latent_representation + eps

        # Decode original and perturbed latent representations
        decoder_out = self.decoder(latent_representation, node_mask)
        perturbed_decoder_out = self.decoder(perturbed_latent_representation, node_mask)






        # NOTE: loss computation will be done in lightning module
        return {'decoder_out': decoder_out,'perturbed_decoder_out': perturbed_decoder_out,'mu': mu, 'log_var': log_var, 'latent_representation': latent_representation}

    def compute_loss(self, moler_output,perturbed_decoder_out, batch):
        x, adj = batch[0], batch[1]
        node_mask = create_node_mask(x)

        adj_update = update_adj_matrix(adj, node_mask)
        one_hot_adj = one_hot_encode_adj(adj_update, 4)
        # adj = mask_adjs(adj, node_mask)
        # recon_loss_classify = F.cross_entropy(pred_graph_argmax.X, X_argmax)+F.cross_entropy(pred_graph_argmax.E, E_argmax)
        loss = self.train_loss(masked_pred_X=moler_output.X, masked_pred_E=moler_output.E,
                               true_X=x, true_E=one_hot_adj)
        loss_pertubed=self.train_loss(masked_pred_X=perturbed_decoder_out.X, masked_pred_E=perturbed_decoder_out.E,
                               true_X=x, true_E=one_hot_adj)







        loss_all=loss+loss_pertubed
        # loss_all=loss
        print(loss_all)
        return loss_all





    def step(self, batch):
        moler_output = self._run_step(batch)

        loss_metrics = {}
        loss_metrics["decoder_loss"] = self.compute_loss(
            moler_output=moler_output['decoder_out'],perturbed_decoder_out=moler_output['perturbed_decoder_out'], batch=batch
        )

        if self._use_clamp_log_var:
            moler_output['log_var'] = torch.clamp(moler_output['log_var'], min=-5, max=5)
        kld_summand = torch.square(moler_output['mu'])
        +torch.exp(moler_output['log_var'])
        -moler_output['log_var']
        -1
        loss_metrics["kld_loss"] = torch.mean(kld_summand) / 2.0
        # loss_metrics['kld_loss'] = torch.distributions.kl_divergence(
        #     moler_output.q, moler_output.p
        # ).mean()
        # kld weight will start from 0 and increase to the original amount.

        annealing_factor = (
            self.trainer.global_step % (self._num_train_batches // 4)
            if self._using_cyclical_anneal
            else self.trainer.global_step
        )

        loss_metrics[
            "kld_weight"
        ] = (  # cyclical anealing where each cycle will span 1/4 of the training epoch
            1.0 - self._kl_divergence_annealing_beta**annealing_factor
        ) * self._kl_divergence_weight

        loss_metrics["kld_loss"] *= loss_metrics["kld_weight"]

        loss_metrics["loss"] = sum(loss_metrics.values())

        logs = loss_metrics
        return loss_metrics["loss"], logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        for metric in logs:
            self.log(f"train_{metric}", logs[metric], batch_size=self._batch_size)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        for metric in logs:
            self.log(f"val_{metric}", logs[metric], batch_size=self._batch_size)

        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-4
        )

        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self._training_hyperparams["max_lr"],
        #     betas=(0.9, 0.999),
        # )
        if self._use_oclr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self._training_hyperparams["max_lr"],
                div_factor=self._training_hyperparams["div_factor"],
                three_phase=self._training_hyperparams["three_phase"],
                epochs=self.trainer.max_epochs,
                # number of times step() is called by the scheduler per epoch
                # take the number of batches // frequency of calling the scheduler
                steps_per_epoch=self._num_train_batches // self.trainer.max_epochs,
            )

            lr_scheduler_params = {}
            lr_scheduler_params["scheduler"] = lr_scheduler

            lr_scheduler_params["interval"] = "step"
            frequency_of_lr_scheduler_step = self.trainer.max_epochs
            lr_scheduler_params[
                "frequency"
            ] = frequency_of_lr_scheduler_step  # number of batches to wait before calling lr_scheduler.step()

            optimizer_dict = {}
            optimizer_dict["optimizer"] = optimizer
            optimizer_dict["lr_scheduler"] = lr_scheduler_params
            return optimizer_dict
        else:
            return optimizer
