import sys
import itertools
from pytorch_lightning import LightningModule

from rdkit.Chem import Draw
from rdkit import Chem

import torch
import numpy as np

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


class BaseModel_ae(AbstractModel):
    def __init__(self, params, epoch_every,data_info, hidden_size, num_heads, depth, using_lincs, use_clamp_log_var=True):
        """Params is a nested dictionary with the relevant parameters."""
        super(BaseModel_ae, self).__init__()
        # self._init_params(params, dataset)
        self.data_info=data_info
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

        self.encoder = DiffEncoder(max_n_nodes=data_info.max_node_num, Xdim=data_info.max_feat_num,
                                   Edim=data_info.edge_feat, hidden_size=hidden_size, depth=depth, num_heads=num_heads)

        self.decoder = DiffDecoder(max_n_nodes=data_info.max_node_num, Xdim=data_info.max_feat_num,
                                   Edim=data_info.edge_feat, hidden_size=hidden_size, depth=depth, num_heads=num_heads)


        self._include_property_regressors = "graph_properties" in self._params
        self.train_loss = TrainLossDiscrete([5, 0])
        self._batch_size = 512
        self.max_n_nodes = data_info.max_node_num
        # params for latent space

        # If using lincs gene expression
        self._using_lincs = using_lincs
        self.initial_noise_loops = 0
        self.epoch_every=epoch_every

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

        x, adj = batch[0], batch[1]

        node_mask = create_node_mask(x)

        adj_update = update_adj_matrix(adj, node_mask)
        one_hot_adj = one_hot_encode_adj(adj_update, self.data_info.edge_feat)
        Xn,En=x,one_hot_adj
        noise_loops = self.initial_noise_loops + (self.current_epoch // self.epoch_every)
        for _ in range(noise_loops):
            X_noisy = Xn * 0.8 + 0.2 * torch.randn_like(Xn)
            E_noisy = En * 0.8 + 0.2 * torch.randn_like(En)
            E_noisy = 1 / 2 * (E_noisy + torch.transpose(E_noisy, 1, 2))

            Xn = X_noisy
            En = E_noisy

        print(noise_loops)
        Xs = Xn
        Es = En

        Xt = Xs * 0.8 + 0.2 * torch.randn_like(Xs)
        Et = Es * 0.8 + 0.2 * torch.randn_like(Es)
        Et = 1 / 2 * (Et + torch.transpose(Et, 1, 2))


        # Obtain graph level representation of the partial graph

        input_molecule_representations = self.encoder(Xt, Et, node_mask).z
        # Apply latent sampling strategy


        decoder_out = self.decoder(input_molecule_representations, node_mask)

        # NOTE: loss computation will be done in lightning module
        return {'decoder_out': decoder_out,'X':Xs,'E':Es}

    def compute_loss(self, moler_output, X1, E1, batch):

        # recon_loss_classify = F.cross_entropy(pred_graph_argmax.X, X_argmax)+F.cross_entropy(pred_graph_argmax.E, E_argmax)
        loss = self.train_loss(masked_pred_X=moler_output.X, masked_pred_E=moler_output.E,
                               true_X=X1, true_E=E1)

        return loss

    def step(self, batch):
        moler_output = self._run_step(batch)

        loss_metrics = {}
        loss_metrics["decoder_loss"] = self.compute_loss(
            moler_output=moler_output['decoder_out'], X1=moler_output['X'], E1=moler_output['E'], batch=batch
        )

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
            self.parameters(), lr=self._training_hyperparams["max_lr"],weight_decay=1.0e-02
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
