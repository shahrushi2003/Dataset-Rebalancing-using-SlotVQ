import torch.nn as nn
import torch
import torch.nn.functional as F
import lightning
from slot_collector import Slot_Collector

import os
os.chdir("/content/vqtorch_folder")
# Finally importing the module
import vqtorch
# Resetting the current working directory
os.chdir("/content")

from edecoder import Encoder, Decoder
from slot_att import SlotAttention


"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots,
                 num_iterations, hid_dim, codebook_size,
                 beta, use_kmeans, z_norm, cb_norm,
                 affine_lr, sync_nu, replace_freq, device):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.hid_dim = hid_dim
        self.codebook_size = codebook_size
        self.beta = beta
        self.use_kmeans = use_kmeans
        self.hid_dim
        self.z_norm=z_norm
        self.cb_norm=cb_norm
        self.affine_lr=affine_lr
        self.sync_nu=sync_nu
        self.replace_freq=replace_freq
        self.device = device

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim).to(device)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution).to(device)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.hid_dim,
            iters = self.num_iterations,
            eps = 1e-8,
            hidden_dim = 128)

        self.vector_quantizers = []
        for i in range(self.num_slots):
            self.vector_quantizers.append((vqtorch.nn.VectorQuant(feature_size=self.hid_dim,     # feature dimension corresponding to the vectors
                                                                  num_codes=self.codebook_size,         # number of codebook vectors
                                                                  beta=self.beta,            # (default: 0.9) commitment trade-off
                                                                  kmeans_init=self.use_kmeans,    # (default: False) whether to use kmeans++ init
                                                                  norm=self.z_norm,           # (default: None) normalization for the input vectors
                                                                  cb_norm=self.cb_norm,        # (default: None) normalization for codebook vectors
                                                                  affine_lr=self.affine_lr,      # (default: 0.0) lr scale for affine parameters
                                                                  sync_nu=self.sync_nu,         # (default: 0.0) codebook synchronization contribution
                                                                  replace_freq=self.replace_freq,     # (default: None) frequency to replace dead codes
                                                                  dim=-1,              # (default: -1) dimension to be quantized
                                                                  )).to(self.device))
            
        self.vector_quantizers = nn.ModuleList(self.vector_quantizers)

    def warmup_quantizers(self):
        for vq_layer in self.vector_quantizers:
            with torch.no_grad():
              z_e = torch.randn(64, 1, 32).to(self.device)
              output = vq_layer(z_e)
              del output

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(self.device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        quantized_slots = []
        vq_loss = 0
        all_perplexity = []
        final_codes = []
        for s in range(self.num_slots):
          # print("Before Quantization", slots[:, s, :].unsqueeze(1).shape)
          q_slot, vq_dict = self.vector_quantizers[s](slots[:, s, :].unsqueeze(1))
          # print("After Quantization", q_slot.shape)
          curr_vq_loss = vq_dict["loss"]
          vq_loss += curr_vq_loss
          codes = vq_dict["q"].reshape(-1, 1)
          e_mean = F.one_hot(vq_dict["q"], num_classes=self.codebook_size).view(-1, self.codebook_size).float().mean(0)
          perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
          final_codes.append(codes)
          all_perplexity.append(perplexity.item())
          # engagement.append(torch.sum(torch.flatten(vq_dict["q"])).item())
          quantized_slots.append(q_slot)

        slots = torch.cat(quantized_slots, dim=1)
        final_codes = torch.cat(final_codes, dim=1)
        # print("After Concatination", quantized_slots.shape)

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots, vq_loss, all_perplexity, final_codes
    
    
class Lightning_AE(lightning.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.example_input_array = torch.Tensor(32, 3, 28, 28)
        self.opt = opt
        self.alpha = opt["alpha"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SlotAttentionAutoEncoder(resolution = opt["resolution"],
                                              num_slots = opt["num_slots"],
                                              num_iterations = opt["num_iterations"],
                                              hid_dim = opt["hid_dim"],
                                              codebook_size = opt["codebook_size"],
                                              beta = opt["beta"],
                                              use_kmeans = opt["use_kmeans"],
                                              z_norm = opt["z_norm"],
                                              cb_norm = opt["cb_norm"],
                                              affine_lr = opt["affine_lr"],
                                              sync_nu = opt["sync_nu"],
                                              replace_freq = opt["replace_freq"],
                                              device = device).to(device)

        self.training_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        recon_combined, recons, masks, slots, vq_loss, perplexity, codes = self.model(x)
        return recon_combined

    def training_step(self, batch, batch_idx):
        images, paths, labels = batch
        recon_combined, recons, masks, slots, vq_loss, perplexity, codes = self.model(images)
        recon_loss = nn.functional.mse_loss(recon_combined, images)
        loss = self.alpha*recon_loss + vq_loss
        self.log("recon_train_loss", recon_loss, batch_size=self.opt['batch_size'])
        self.log("vq_train_loss", vq_loss, batch_size=self.opt['batch_size'])
        self.log("total_train_loss", loss, batch_size=self.opt['batch_size'])
        # self.log("perplexity", perplexity, batch_size=self.opt['batch_size'])
        self.log("engagement_0", perplexity[0], batch_size=self.opt['batch_size'])
        self.log("engagement_1", perplexity[1], batch_size=self.opt['batch_size'])
        self.log("engagement_2", perplexity[2], batch_size=self.opt['batch_size'])
        self.training_step_outputs.append((paths, labels, codes))
        del slots, masks
        return loss

    def on_train_epoch_end(self):
        coll = Slot_Collector(num_slots = self.opt["num_slots"], codebook_size = self.opt["codebook_size"])
        for idx, (paths, target, code) in enumerate(self.training_step_outputs):
            target = target.to(self.device)
            coll.collect(code.cpu(), paths, target)

        print("Slots Distribution for Epoch:", self.current_epoch)
        print([len(coll.slot_collector[i]) for i in coll.slot_collector])

        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        images, _, labels = batch
        recon_combined, recons, masks, slots, vq_loss, perplexity, codes = self.model(images)
        recon_loss = nn.functional.mse_loss(recon_combined, images)
        loss = self.alpha*recon_loss + vq_loss
        self.log("recon_val_loss", recon_loss, batch_size=self.opt['batch_size'])
        self.log("vq_val_loss", vq_loss, batch_size=self.opt['batch_size'])
        self.log("total_val_loss", loss, batch_size=self.opt['batch_size'])
        # self.log("perplexity", perplexity, batch_size=self.opt['batch_size'])
        self.log("engagement_0", perplexity[0], batch_size=self.opt['batch_size'])
        self.log("engagement_1", perplexity[1], batch_size=self.opt['batch_size'])
        self.log("engagement_2", perplexity[2], batch_size=self.opt['batch_size'])
        del slots, masks
        return loss

    def test_step(self, batch, batch_idx):
        images, _, labels = batch
        recon_combined, recons, masks, slots, vq_loss, perplexity, codes = self.model(images)
        recon_loss = nn.functional.mse_loss(recon_combined, images)
        loss = self.alpha*recon_loss + vq_loss
        self.log("recon_test_loss", recon_loss, batch_size=self.opt['batch_size'])
        self.log("vq_test_loss", vq_loss, batch_size=self.opt['batch_size'])
        self.log("total_test_loss", loss, batch_size=self.opt['batch_size'])
        # self.log("perplexity", perplexity, batch_size=self.opt['batch_size'])
        self.log("engagement_0", perplexity[0], batch_size=self.opt['batch_size'])
        self.log("engagement_1", perplexity[1], batch_size=self.opt['batch_size'])
        self.log("engagement_2", perplexity[2], batch_size=self.opt['batch_size'])
        del slots, masks
        return loss

    def predict_step(self, batch, batch_idx):
        images, _, labels = batch
        recon_combined, recons, masks, slots, vq_loss, perplexity, codes = self.model(images)
        return recon_combined, recons, masks, slots, vq_loss, perplexity, codes

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opt["learning_rate"])