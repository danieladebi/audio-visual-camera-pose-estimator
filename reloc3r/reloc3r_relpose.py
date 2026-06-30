from copy import deepcopy
import os
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from functools import partial
import reloc3r.utils.path_to_croco
from reloc3r.patch_embed import ManyAR_PatchEmbed
from models.pos_embed import RoPE2D 
from models.blocks import Block, DecoderBlock
from reloc3r.pose_head import PoseHead
from reloc3r.utils.misc import freeze_all_params, transpose_to_landscape
from pdb import set_trace as bb
from huggingface_hub import PyTorchModelHubMixin
import os
import torch
from torchprofile import profile_macs

import torchvision.models as models
import torch.nn.functional as F
import kornia.augmentation as K
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pysofaconventions

# from models import *


# parts of the code adapted from 
# 'https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/croco.py#L21'
# 'https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/model.py#L46'
class Reloc3rRelpose(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=512,    # 512 input image size
                 patch_size=16,         # patch_size 
                 enc_embed_dim=1024,    # 1024 encoder feature dimension
                 enc_depth=24,          # encoder depth 
                 enc_num_heads=16,      # encoder number of heads in the transformer block 
                 dec_embed_dim=768,     # decoder feature dimension 
                 dec_depth=12,          # decoder depth 
                 dec_num_heads=12,      # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',   # positional embedding (either cosine or RoPE100)
                 #audio_enc_depth=24,
                # audio_dec_depth=12,
                 in_channels=3,
               #  audio_channels=7,
                 has_audio=False,
                 has_audio_embedding=False,
                 is_audio = False,
                 #audio_only_model=True
                 is_slfm=False,
                 use_doa=False,
                 use_doa_and_embed=False,
                 use_detr_cross_attention=False,
                 has_imu=False
                ):   
        super(Reloc3rRelpose, self).__init__()

       # self.audio_only_model = audio_only_model
        self.has_audio_embedding = has_audio_embedding
        # patchify and positional embedding
        self.in_channels = 7 if is_audio else in_channels # 2 for binaural 7 otherwise
        self.is_slfm = is_slfm
        if (is_slfm or in_channels == 2) and is_audio:
            self.in_channels = 2
        self.patch_embed = ManyAR_PatchEmbed(img_size, patch_size, self.in_channels, enc_embed_dim) # default 3
        self.pos_embed = pos_embed
        self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        self.rope = RoPE2D(freq=freq)
        self.has_audio = has_audio

        # ViT encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim 
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        
    
        # Support 7-channel spectrograms for audio input
        # TODO: finish this implementation
        # self.audio_in_chans = audio_channels
        # self.audio_depth = audio_enc_depth
        self.audio_embed_dim = enc_embed_dim 
        # self.audio_patch_embed = ManyAR_PatchEmbed(img_size, patch_size, self.audio_in_chans, enc_embed_dim)
        # self.audio_blocks = nn.ModuleList([
        #     Block(enc_embed_dim, enc_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
        #     for i in range(audio_enc_depth)])
        # self.audio_norm = norm_layer(enc_embed_dim)

        # ViT decoder
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)  # transfer from encoder to decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)

        # ViT audio decoder
        # self.audio_decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)  # transfer from audio encoder to decoder
        # self.audio_dec_blocks = nn.ModuleList([
        #     DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
        #     for i in range(audio_dec_depth)]) 
        # self.audio_dec_norm = norm_layer(dec_embed_dim)

        # # linear audio embedding model (sequence of multiple linear layers)
        audio_embed_proj_input = 1024
        if use_doa: 
            audio_embed_proj_input = 360
        if use_doa_and_embed:
            audio_embed_proj_input = 1024 + 360
        self.audio_embed_proj = nn.Sequential(
            nn.Linear(audio_embed_proj_input, self.audio_embed_dim),
            nn.LayerNorm(self.audio_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.audio_embed_dim, self.dec_embed_dim),
            nn.LayerNorm(self.dec_embed_dim),
            nn.ReLU(inplace=True)
        ) if has_audio else None

        self.use_detr_cross_attention = use_detr_cross_attention
        if self.use_detr_cross_attention:
            self.upsample_per_query = 32  # 8 * 32 = 256

            self.query_token_expander = nn.Linear(dec_embed_dim, dec_embed_dim * self.upsample_per_query, bias=True) 
            self.cross_query_decoder = CrossModalQueryDecoder(
                query_dim=dec_embed_dim,
                num_queries=8,
                nhead=min(8, dec_num_heads),
                num_layers=2,
                mem_dim=dec_embed_dim,
                audio_dim=self.dec_embed_dim if has_audio else None
            )
        self.has_imu = has_imu
        imu_size = 13
        if self.has_imu:
            self.imu_head = nn.Linear(imu_size, enc_embed_dim) 
        if self.has_imu and self.has_audio:
            self.imu_audio_head = nn.Linear(imu_size+audio_embed_proj_input, audio_embed_proj_input)

        # pose regression head
        self.pose_head = PoseHead(net=self, has_audio=self.has_audio, is_slfm=self.is_slfm, use_cross_attn=self.use_detr_cross_attention, has_imu=self.has_imu)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

        self.initialize_weights() 

    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_encoder(self):
        freeze_all_params([self.patch_embed, self.enc_blocks])

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
   
        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    # TODO: implement this properly
    # def _encode_audio(self, audio_spec, true_shape):
    #     # embed the audio into patches  (x has size B x Npatches x C)
    #     x, pos = self.audio_patch_embed(audio_spec, true_shape=true_shape)

    #     # add positional embedding without cls token
    #     assert self.enc_pos_embed is None

    #     # now apply the transformer encoder and normalization
    #     for blk in self.audio_blocks:
    #         x = blk(x, pos)

    #     x = self.audio_norm(x)
    #     return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encoder(self, view1, view2):
        if self.has_audio and not self.has_audio_embedding and "audio_spec" in view1:
            img1 = view1['audio_spec']
            img2 = view2["audio_spec"]
        else:
            img1 = view1['img']
            img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)
        # Audio encoding (if present)
        # if 'audio_spec' in view1 and 'audio_spec' in view2:
        #     audio1 = view1['audio_spec']
        #     audio2 = view2['audio_spec']
        #     audio_shape1 = view1.get('audio_true_shape', torch.tensor(audio1.shape[-2:])[None].repeat(B, 1))
        #     audio_shape2 = view2.get('audio_true_shape', torch.tensor(audio2.shape[-2:])[None].repeat(B, 1))
        #     audio_feat1, audio_pos1, _ = self._encode_audio(audio1.float().to(device=img1.device), audio_shape1)
        #     audio_feat2, audio_pos2, _ = self._encode_audio(audio2.float().to(device=img2.device), audio_shape2)
        #     return (shape1, shape2, audio_shape1, audio_shape2), (feat1, feat2, audio_feat1, audio_feat2), (pos1, pos2, audio_pos1, audio_pos2)
        #     # You can further process or fuse audio_feat1/audio_feat2 as needed
        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        """
        Decoder for image features, optionally with external audio embeddings.
        If embedding1/embedding2 are provided, concatenate them to f1/f2 along the sequence (patch) dimension.
        """
        # If audio embeddings are provided, concatenate them to the features
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk in self.dec_blocks:
            # img1 side
            f1, _ = blk(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)
   
    
    def _downstream_head(self, decout, img_shape, audio_embedding=None):
        if self.use_detr_cross_attention:
            # take the last decoder output as image memory and pass audio embedding if available
            # tokens = decout[-1]  # (B, S, D)
            return self.head([decout], img_shape)
        elif audio_embedding is None:
            B, S, D = decout[-1].shape
            return self.head(decout, img_shape)
        else:

            B, S, D = decout[-1].shape
            return self.head(decout, img_shape, audio_embedding)
        # else:
        #     # Handle both image and audio features
        #     B, S, D = decout[-1].shape
                        
        #     # Concatenate image and audio features along the sequence dimension
        #   #  combined_decout = torch.cat(decout, dim=1)
        #     # Ensure both shapes are on the same device
        #     combined_shape = img_shape 
        #     # Apply the pose head to the combined features
        #     return self.head(decout, combined_shape)

    def forward(self, view1, view2, return_embedding=False):
        encoder_outputs = self._encoder(view1, view2)  # Handles both audio and non-audio cases


        if "audio_embedding" in view1 and "imu" in view2:
            audio_embedding1 = view1["audio_embedding"].to(device=view1['img'].device)
            audio_embedding2 = view2["audio_embedding"].to(device=view2['img'].device)
            
            imu1 = view1["imu"].to(device=view1['img'].device).float()
            imu2 = view2["imu"].to(device=view2['img'].device).float()
            
            concat_embed1 = torch.cat((audio_embedding1, imu1.float()), dim=1)
            concat_embed2 = torch.cat((audio_embedding2, imu2.float()), dim=1)
            
            embedding1 = self.imu_audio_head(concat_embed1).float()
            embedding2 = self.imu_audio_head(concat_embed2).float()
        elif "audio_embedding" in view1 and "audio_embedding" in view2:
            # If audio embeddings are provided, concatenate them to the decoder outputs
            embedding1 = view1["audio_embedding"].to(device=view1['img'].device)
            embedding2 = view2["audio_embedding"].to(device=view2['img'].device)
            # Ensure embeddings are on the same device and have correct dtype
            # Pass embeddings through the audio embedding projection model
            embedding1 = self.audio_embed_proj(embedding1).float()
            embedding2 = self.audio_embed_proj(embedding2).float()
            # Concatenate embeddings to the decoder outputs along the sequence (patch) dimension
        elif "imu" in view1 and "imu" in view2:
            #if self.has_imu:
            imu1 = view1["imu"].to(device=view1['img'].device)
            imu2 = view2["imu"].to(device=view2['img'].device)
            embedding1 = self.imu_head(imu1.float()).float()
            embedding2 = self.imu_head(imu2.float()).float()
        else:
            embedding1, embedding2 = None, None

        # Unpack encoder outputs based on presence of audio
        # if len(encoder_outputs[1]) == 4:
        #     # Audio features present
        #     (shape1, shape2, audio_shape1, audio_shape2), (feat1, feat2, audio_feat1, audio_feat2), (pos1, pos2, audio_pos1, audio_pos2) = encoder_outputs
        #     # Process combined image and audio features through the correct audio decoder
        #     dec1_combined, dec2_combined = self._decoder_audio(
        #         feat1, pos1, feat2, pos2, audio_feat1, audio_pos1, audio_feat2, audio_pos2
        #     )
        # else:
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = encoder_outputs
        dec1_combined, dec2_combined = self._decoder(feat1, pos1, feat2, pos2)
        
        if hasattr(self, "cross_query_decoder") and self.use_detr_cross_attention:

            dec_out1 = [tok.float() for tok in dec1_combined]
            dec_out2 = [tok.float() for tok in dec2_combined]
        
            final_tokens1 = dec_out1[-1]
            final_tokens2 = dec_out2[-1]
            
            q_out1 = self.cross_query_decoder(final_tokens1, audio_embedding=embedding1)
            q_out2 = self.cross_query_decoder(final_tokens2, audio_embedding=embedding2)

            # Expand cross-query outputs from (B, 8, 768) -> (B, 256, 768)
            B, nq, D = q_out1.shape
            assert nq * self.upsample_per_query == 256, "Expected target query count 256"
            

            # apply to q_out1
            q1_exp = self.query_token_expander(q_out1.view(-1, D))               # (B*nq, D*upsample)
            q1_exp = q1_exp.view(B, nq, self.upsample_per_query, D).contiguous()     # (B, nq, upsample, D)
            q_out1 = q1_exp.view(B, nq * self.upsample_per_query, D)                 # (B, 256, D)

            # apply to q_out2
            q2_exp = self.query_token_expander(q_out2.view(-1, D))
            q2_exp = q2_exp.view(B, nq, self.upsample_per_query, D).contiguous()
            q_out2 = q2_exp.view(B, nq * self.upsample_per_query, D)
            
            dec_out1 = q_out1.float()
            dec_out2 = q_out2.float()
            
            # Do not append the small set of cross-query tokens to the decoder outputs list,
            # as that would make dec_out[-1] refer to the queries (num_queries) instead of the image memory (S),
            # causing a mismatch (e.g. 1024 vs 32 channels/tokens) in the downstream head.
            # Keep q_out1/q_out2 available separately if needed.
            
            with torch.cuda.amp.autocast( enabled=False):
                pose1 = self._downstream_head(dec_out1, shape1, audio_embedding=embedding1)
                pose2 = self._downstream_head(dec_out2, shape2, audio_embedding=embedding2)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                pose1 = self._downstream_head([tok.float() for tok in dec1_combined], shape1, audio_embedding=embedding1)
                pose2 = self._downstream_head([tok.float() for tok in dec2_combined], shape2, audio_embedding=embedding2)

        if return_embedding:
            # dec1_combined and dec2_combined are iterators of all decoder layer outputs, last is the final embedding
            # We want the last layer's embedding before pose prediction
            return pose1, pose2, pose1["embedding"], pose2["embedding"]

        return pose1, pose2

class CrossModalQueryDecoder(nn.Module):
    def __init__(self, query_dim, num_queries=32, nhead=8, num_layers=2, mem_dim=None, audio_dim=None, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.query_embed = nn.Parameter(torch.randn(num_queries, query_dim))
        self.proj_mem = nn.Linear(mem_dim if mem_dim is not None else query_dim, query_dim)
        # audio projection (optional)
        self.proj_audio = nn.Linear(audio_dim if audio_dim is not None else query_dim, query_dim) if audio_dim is not None else None

        layer = nn.TransformerDecoderLayer(d_model=query_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, memory, audio_embedding=None):
        # memory: (B, S, D_mem)
        B = memory.shape[0]
        mem = self.proj_mem(memory)                 # (B, S, query_dim)
        mem = mem.transpose(0, 1)                   # (S, B, query_dim) for nn.Transformer*

        if audio_embedding is not None and self.proj_audio is not None:
            a = self.proj_audio(audio_embedding)    # (B, query_dim) or (B, T, query_dim)
            # make audio token(s) have shape (T, B, query_dim) to append along sequence dim (S -> S+T)
            if a.dim() == 2:
                # single audio token per batch element -> (1, B, query_dim)
                mem = torch.cat((mem, a.unsqueeze(0)), dim=0)  # (S+1, B, query_dim)
            elif a.dim() == 3:
                # multiple audio tokens per batch element -> (T, B, query_dim)
                mem = torch.cat((mem, a.transpose(0, 1)), dim=0)  # (S+T, B, query_dim)
            else:
                raise ValueError(f"Unsupported audio projection dimensions: {a.shape}")

        # prepare learnable queries: (num_queries, B, query_dim)
        queries = self.query_embed.unsqueeze(1).expand(-1, B, -1).contiguous()

        out = self.decoder(tgt=queries, memory=mem)    # (num_queries, B, query_dim)
        out = out.transpose(0, 1).contiguous()         # (B, num_queries, query_dim)
        return out

class PolicyClassificationModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_dim=1024*3, arch="resnet", model_type=50, dropout=0.2):
        super().__init__()

        self.arch = arch
        if arch == "mlp":        
            self.fc1 = nn.Linear(input_dim, 1024)
            self.bn1 = nn.BatchNorm1d(1024)
            
            self.fc2 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(512)
            
            self.fc3 = nn.Linear(512, 128)
            self.bn3 = nn.BatchNorm1d(128)
            
            self.fc_out = nn.Linear(128, 1)  # Binary classification
            
            self.dropout = nn.Dropout(dropout)
        else:
            if model_type == 18:
                self.backbone = models.resnet18(weights=None)  # resnet18 # no pretrained weights
                self.fc_dim = 512
            elif model_type == 50:
                self.backbone = models.resnet50(weights=None)  # resnet50 # no pretrained weights
                self.fc_dim = 2048
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            self.backbone.conv1 = nn.Conv2d(
                in_channels=7,  # 7 spectrogram channels (1 per mic)
                out_channels=64,
                kernel_size=(7,7),
                stride=(2,2),
                padding=(3,3),
                bias=False
            )
            # Single logit for binary classification (use with BCEWithLogitsLoss); apply torch.sigmoid in inference if you need probabilities.
            self.backbone.fc = nn.Linear(self.fc_dim, 1)  # 2048 for resnet50, 512 for resnet18

    def forward(self, view1, view2):
        # Accept either a raw tensor or a dict containing 'audio_spec'
        if self.arch == "mlp":
            emb1 = view1["yamnet_embedding"].to(device=view1['policy_decision'].device)
            emb2 = view2["yamnet_embedding"].to(device=view2['policy_decision'].device)
            x = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)], dim=1)
            x = self.dropout(F.relu(self.bn1(self.fc1(x))))
            x = self.dropout(F.relu(self.bn2(self.fc2(x))))
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.fc_out(x)
            return x
        else:
            x = view1.get('audio_spec', view1.get('img', None))
            if x is None:
                raise KeyError("Input dict must contain 'audio_spec' or 'img'.")
            logits = self.backbone(x)  # shape (B, 1)
            # # Extract penultimate features as embedding
            # feat = self.backbone.conv1(x)
            # feat = self.backbone.bn1(feat)
            # feat = self.backbone.relu(feat)
            # feat = self.backbone.maxpool(feat)
            # feat = self.backbone.layer1(feat)
            # feat = self.backbone.layer2(feat)
            # feat = self.backbone.layer3(feat)
            # feat = self.backbone.layer4(feat)
            # feat = self.backbone.avgpool(feat)
            # embedding = torch.flatten(feat, 1)

            # # Optionally return embedding as well if requested
            # ret_flag = False
            # v = view1.get('return_embedding', None)
            # if v is not None:
            #     ret_flag = bool(v.flatten()[0].item()) if isinstance(v, torch.Tensor) else bool(v)
            # if not ret_flag:
            #     v = view1.get('save_embedding', None)
            #     if v is not None:
            #         ret_flag = bool(v.flatten()[0].item()) if isinstance(v, torch.Tensor) else bool(v)

            # if ret_flag:
            #     logits = (logits, embedding)
            return logits



class AudioDOAModel(nn.Module, PyTorchModelHubMixin):   
    def __init__(self, n_degs=180, embed_dim=512, model_type=18, is_slfm=False): # embed_dim 512 for resnet18
        super().__init__()

        self.fc_dim = 512
        if model_type == 18:
            self.backbone = models.resnet18(weights=None) # resnet18 # no pretrained weights
        elif model_type == 50:
            self.fc_dim = 2048
            self.backbone = models.resnet50(weights=None) # resnet50 # no pretrained weights
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2 if is_slfm else 7,  # 7 spectrogram channels (1 per mic)
            out_channels=64,
            kernel_size=(7,7),
            stride=(2,2),
            padding=(3,3),
            bias=False
        )
        self.backbone.fc = nn.Linear(self.fc_dim, embed_dim) # 2048 for resnet50, 512 for resnet18
        
        self.embed_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.reg_head = nn.Linear(embed_dim, n_degs)


    def forward(self, x, return_embedding=True):  # x: (B, [7 or 2], F, T)

        x = x["audio_spec"]

        feats = self.backbone(x)
        emb = self.embed_head(feats)
        doa_outputs = self.reg_head(emb)
        
        doa_outputs = torch.nan_to_num(doa_outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        doa_outputs = (doa_outputs - doa_outputs.min(dim=1, keepdim=True)[0]) / (doa_outputs.max(dim=1, keepdim=True)[0] - doa_outputs.min(dim=1, keepdim=True)[0] + 1e-8)
      
        if return_embedding:
            return doa_outputs, emb
        return doa_outputs

class DOACameraPoseModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_embed_dim=1024, num_layers=20):
        super(DOACameraPoseModel, self).__init__()

        # Simple MLP that maps an input audio embedding to a pose embedding,
        # then predicts translation (3) and a 9D rotation representation.
        hidden_dim = 1024*2

        mlp_layers = []
        for i in range(num_layers):
            in_dim = input_embed_dim if i == 0 else hidden_dim
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*mlp_layers)

        # Heads that produce translation (3) and a 9D rotation representation
        self.fc_t = nn.Linear(hidden_dim, 3)
        self.fc_rot = nn.Linear(hidden_dim, 9)

        # # Lightweight initialization
        # for m in self.mlp:
        #     if isinstance(m, nn.Linear):
        #     nn.init.xavier_uniform_(m.weight)
        #     if m.bias is not None:
        #         nn.init.constant_(m.bias, 0.)
        # nn.init.xavier_uniform_(self.fc_t.weight)
        # nn.init.constant_(self.fc_t.bias, 0.)
        # nn.init.xavier_uniform_(self.fc_rot.weight)
        # nn.init.constant_(self.fc_rot.bias, 0.)

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """

        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)).float())
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
   
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def forward(self, view1, view2):
        if "input_doa" in view1 and "input_doa" in view2:
            input_1 = view1["input_doa"].squeeze(1)
            input_2 = view2["input_doa"].squeeze(1)
        else:
            input_1 = view1["audio_embedding"].squeeze(1)
            input_2 = view2["audio_embedding"].squeeze(1)

        B = input_1.shape[0]  # batch size
        device = input_1.device

        output1 = self.mlp(input_1)   
        output2 = self.mlp(input_2)

        # if self.output_size == 1:
        #     out_r1 = self.fc_rot(output1)
        #     out_r2 = self.fc_rot(output2)

        #     pose1 = {"pose": out_r1}
        #     pose2 = {"pose": out_r2}
    # else:
        out_t1 = self.fc_t(output1).float()
        out_t2 = self.fc_t(output2).float()

        out_r1 = self.fc_rot(output1).float()
        out_r2 = self.fc_rot(output2).float()

        pose1 = {"pose": self.convert_pose_to_4x4(B, out_r1, out_t1, device)}
        pose2 = {"pose": self.convert_pose_to_4x4(B, out_r2, out_t2, device)}

        return pose1, pose2


class AudioClassPolicyModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super(AudioClassPolicyModel, self).__init__()
        # TODO: finish writing model
        # NOTE: inputs should be audio classes ONLY, output should be policy selection (audio or visual)
        


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.LeakyReLU(inplace=True),
    )


class CameraEncoder(nn.Module):
    def __init__(self):
        super(CameraEncoder, self).__init__()
        self.img_augmentation = K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, same_on_batch=False, p=0.9) if True else nn.Identity()
        self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def transform_img(self, imgs, augment=False):
        # import pdb; pdb.set_trace()
        if augment:
            imgs = self.img_augmentation(imgs)
        imgs = self.img_transform(imgs).detach()
        return imgs

class CameraATTNNet(CameraEncoder):
    def __init__(self):
        super(CameraATTNNet, self).__init__()
        backbone = 'resnet18'
        imagenet_pretrain = True
        weights = "IMAGENET1K_V1" if imagenet_pretrain else None 
        if backbone == 'resnet18':
            self.backbone = self.get_truncated_resnet(torchvision.models.resnet18(weights=weights))
        elif backbone == 'resnet34':
            self.backbone = self.get_truncated_resnet(torchvision.models.resnet34(weights=weights))
        elif backbone == 'resnet50':
            self.backbone = self.get_truncated_resnet(torchvision.models.resnet50(weights=weights))
        else:
            raise NotImplementedError
        
        backbone_downsample_rate = 16
        img_size= 256
        attn_in_channels = int((img_size / backbone_downsample_rate) * (img_size / backbone_downsample_rate))

        self.attn_convs = nn.Sequential(
            conv2d(in_channels=attn_in_channels, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )

        attn_conv_downsample_rate = 8
        attn_out_size = int(np.ceil(img_size / backbone_downsample_rate / attn_conv_downsample_rate) * np.ceil(img_size / backbone_downsample_rate / attn_conv_downsample_rate))
        
        visual_feature_dim = 512
        self.linear = nn.Linear(attn_out_size * 128, visual_feature_dim)

        # init weights
        for m in self.attn_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.linear.weight, mean=0.0, std=0.01)

        # if args.add_geometric:
        #     out_dim = 1
        #     self.rot_head = nn.Sequential(
        #         nn.Linear(visual_feature_dim, int(pr.visual_feature_dim // 2)),
        #         nn.ReLU(True),
        #         nn.Linear(int(pr.visual_feature_dim // 2), out_dim)
        #     )
        #     for m in self.rot_head.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
            

    def forward(self, img_1, img_2, augment=False, return_angle=False, backbone=False, correlation=False):
        # import pdb; pdb.set_trace()
        ''' 
            img_1: (N, C, H, W)
            img_2: (N, C, H, W)
        '''
        if backbone: 
            im_feature1 = self.forward_backbone(self, img_1, augment)
            return im_feature1

        if correlation:
            x = self.forward_correlation(img_1, img_2, return_angle=return_angle)
            return x

        img_1 = self.transform_img(img_1, augment)
        img_2 = self.transform_img(img_2, augment)
        im_feature1 = self.backbone(img_1)
        im_feature2 = self.backbone(img_2)
        x = self.forward_correlation(im_feature1, im_feature2, return_angle=return_angle)
        return x

    def forward_backbone(self, img, augment):
        img = self.transform_img(img, augment)
        im_feature = self.backbone(img)
        return im_feature

    def forward_correlation(self, im_feature1, im_feature2, return_angle=False):
        aff = self.compute_corr_softmax(im_feature1, im_feature2)
        x = self.attn_convs(aff)
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        if return_angle:
            pred = self.rot_head(x).squeeze(-1)

            return x, pred
        return x

    def get_truncated_resnet(self, resnet):
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

    def compute_corr_softmax(self, im_feature1, im_feature2):
        _, _, h1, w1 = im_feature1.size()
        _, _, h2, w2 = im_feature2.size()
        im_feature2 = im_feature2.transpose(2, 3)
        im_feature2_vec = im_feature2.contiguous().view(im_feature2.size(0), im_feature2.size(1), -1)
        im_feature2_vec = im_feature2_vec.transpose(1, 2)
        im_feature1_vec = im_feature1.contiguous().view(im_feature1.size(0), im_feature1.size(1), -1)
        corrfeat = torch.matmul(im_feature2_vec, im_feature1_vec)
        corrfeat = corrfeat.view(corrfeat.size(0), h2*w2, h1, w1)
        corrfeat  = F.softmax(corrfeat, dim=1)
        return corrfeat

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, kernel_size=4):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv])



def unet_conv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[downconv, downnorm, downrelu])
    else:
        return nn.Sequential(*[downconv])
    
class CondAudioEncoder(nn.Module):
    def __init__(self, audio_backbone='resnet18', audio_feature_dim=512, n_fft=512, win_length=400):
        super(CondAudioEncoder, self).__init__()

        self.audio_backbone = audio_backbone
        self.audio_feature_dim = audio_feature_dim
        self.net = self.construct_audio_net()
        self.n_fft = n_fft
        self.win_length = win_length
        self.cond_clip_length = 1.0
        self.samp_sr = 16000  # TODO: downsample from 48000
        self.log_offset = 1e-5
        self.use_real_imag = False  

        # if args.add_geometric:
        #     out_dim = 1
        #     self.pred_head = nn.Sequential(
        #         nn.Linear(pr.audio_feature_dim, int(pr.audio_feature_dim // 2)),
        #         nn.ReLU(True),
        #         nn.Linear(int(pr.audio_feature_dim // 2), out_dim)
        #     )
        #     for m in self.pred_head.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, audio, return_angle=False, augment=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio: (N, C, L)
        '''
        # Re-cut the conditional audio clip to meet the requirement
        audio = audio[..., :int(self.cond_clip_length * self.samp_sr)]

        audio = self.wave2spec(audio)
        x = self.net(audio)
        if return_angle: 
            pred = self.pred_head(x).squeeze(-1) 
            return x, pred
        return x
    
    def construct_audio_net(self, audio_backbone='resnet18'):
        in_channels = 4
        if self.audio_backbone == 'resnet10':
            model = torchvision.models.resnet._resnet(torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], weights=None, progress=False)
        elif self.audio_backbone == 'resnet18':
            model = torchvision.models.resnet18(weights=None)
        elif self.audio_backbone == 'resnet34':
            model = torchvision.models.resnet34(weights=None)
        elif self.audio_backbone == 'resnet50':
            model = torchvision.models.resnet50(weights=None)

        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, self.audio_feature_dim)

        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        return model
    
    def wave2spec(self, wave, return_complex=False):
        '''
            return normalized magnitude and phase spectrogram: (N, C, F, T), C = 4 with both magnitude and phase
        '''
        # import pdb; pdb.set_trace()
        N, C, L = wave.shape
        wave = wave.view(N * C, -1)

        frames = 256
        hop_length = int(L // (frames - 1))

        spec = torch.stft(
            input=wave,
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            return_complex=True
        )

        spec = spec.contiguous().view(N, C, *spec.shape[1:])
        if return_complex:
            return spec

        if self.use_real_imag:
            spec = torch.view_as_real(spec)
        else:
            mag, phase = spec.abs().unsqueeze(-1), spec.angle().unsqueeze(-1)
            mag = self.normalize_magnitude(mag)
            phase = self.normalize_phase(phase)
            spec = torch.cat([mag, phase], dim=-1)
        spec = spec.permute(0, 1, 4, 2, 3)
        spec = spec.contiguous().view(N, -1, *spec.shape[3:])
        spec = spec[:, :, :-1, :frames]
        return spec

    def normalize_magnitude(self, spec):
        # import pdb; pdb.set_trace()
        spec_min = -100
        spec_max = 60
        spec = torch.maximum(spec, torch.tensor(self.log_offset))
        spec = 20 * torch.log10(spec)
        spec = (spec - spec_min) / (spec_max - spec_min) * 2 - 1
        spec = torch.clip(spec, -1.0, 1.0)
        return spec

    def normalize_phase(self, phase):
        pi = 3.1416
        phase = phase / pi
        phase = torch.clip(phase, -1.0, 1.0)
        return phase


class AudioCondUNet(nn.Module):
    def __init__(self, ngf=64):
        super(AudioCondUNet, self).__init__()
        #initialize layers
        input_nc = 2
        output_nc = 2
        n_view = 2
        self.no_vision = False
        self.no_cond_audio = False
        self.mono2binaural = True
        
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, outermost=True)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)

        self.audio_visual_feat_dim = ngf * 8
        cond_feat_dim = 0

        audio_feature_dim = 512
        visual_feature_dim = 512
        if self.no_vision and not self.no_cond_audio:
            cond_feat_dim = audio_feature_dim
        elif not self.no_vision and self.no_cond_audio:
            cond_feat_dim = visual_feature_dim
        elif not self.no_vision and not self.no_cond_audio:
            cond_feat_dim = visual_feature_dim + audio_feature_dim
        
        self.audio_visual_feat_dim += cond_feat_dim

        self.audionet_upconvlayer1 = unet_upconv(self.audio_visual_feat_dim, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True)


    def forward(self, input_audio, cond_feats):
        '''
            input_audio: (N, C, F, T), C = 1 or 2
            cond_feats: (N, C), C = X, all the conditional features
        '''
        # import pdb; pdb.set_trace()
        audio_conv1feature = self.audionet_convlayer1(input_audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        audioVisual_feature = audio_conv5feature
        if not self.no_cond_audio or not self.no_vision:
            cond_feats = cond_feats.view(cond_feats.size(0), -1, 1, 1).repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1])
            audioVisual_feature = torch.cat((cond_feats, audioVisual_feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        

        if self.mono2binaural:
            mask_prediction = torch.sigmoid(prediction) * 2 - 1
            spec_diff_real = input_audio[:, 0, :, :-1] * mask_prediction[:, 0, :, :] - input_audio[:, 1, :, :-1] * mask_prediction[:, 1, :, :]
            spec_diff_img = input_audio[:, 0, :, :-1] * mask_prediction[:, 1, :, :] + input_audio[:, 1, :, :-1] * mask_prediction[:, 0, :, :]
            prediction = torch.cat((spec_diff_real.unsqueeze(1), spec_diff_img.unsqueeze(1)), dim=1)
        else:
            prediction = torch.sigmoid(prediction) * 2 - 1
        return prediction



class FloatEmbeddingSine(nn.Module):
    """
    This is a simple version of the float embedding which convert a float number to a high dimension vector.
    It is adopted from PositionEmbeddingSine from paper DETR.
    """
    def __init__(self, num_pos_feats=512, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            self.scale = 2 * np.pi
        else:
            self.scale = scale

    def forward(self, x):
        '''
            x: (N, K)
        '''
        # import pdb; pdb.set_trace()
        x = x * self.scale # we use 1 for scale since it's convert to 2 pi already
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        x = x / dim_t
        x = torch.stack([x[:, 0::2].sin(), x[:, 0::2].cos()], dim=-1).flatten(-2)
        return x

class SLfMNet(nn.Module, PyTorchModelHubMixin):
    # TODO: import SLFMNet libraries
    def __init__(self, n_view=2, n_fft=512, hop_length=100, win_length=400): # hop_length=160
        super(SLfMNet, self).__init__()
        
        self.n_view = n_view
        self.no_vision = False
        self.no_cond_audio = False
        self.add_geometric = False
      #  self.use_gt_rotation = args.use_gt_rotation
        self.generative_loss_ratio = 1. # args.generative_loss_ratio
        self.visual_feature_dim = 512
        self.loss_type = "L1"

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length 
        self.log_offset = 1e-5

        self.vision_net, self.audio_net = CameraATTNNet(), CondAudioEncoder()
        self.generative_net = AudioCondUNet()
        self.mono2binaural = True

        # if self.use_gt_rotation:
        #     self.rota_embedding = FloatEmbeddingSine(num_pos_feats=self.visual_feature_dim, scale=1)

       # self.freeze_param(args)
    

    def forward(self, inputs, loss=False, evaluate=False, inference=False):
        # import pdb; pdb.set_trace()
        augment = not (evaluate or inference)
        cond_audios, audio_input, audio_output = self.generate_audio_pair(inputs)
        cond_feats = self.encode_conditional_feature(inputs, cond_audios, augment)
        # import pdb; pdb.set_trace()
        pred_audio = self.generative_net(audio_input, cond_feats)

        if loss:
            loss = self.calc_loss(inputs, audio_output, pred_audio)
            return loss
        if evaluate:
            output = self.calc_loss(inputs, audio_output, pred_audio, evaluate=True)
            return output, cond_feats
        if inference:
            output = self.inference(inputs, pred_audio)
            return output
        
        return pred_audio
    

    def calc_loss(self, inputs, target_audio, pred_audio, evaluate=False):
        output = {}
        N = pred_audio.shape[0]
        if pred_audio.shape != target_audio.shape:
            target_audio = target_audio[..., :-1]
        spec_weight = 1
        if self.loss_type == 'L1':
            spec_loss = F.l1_loss(pred_audio, target_audio, reduction='none')
            spec_weight = 10
        elif self.loss_type == 'L2':
            spec_loss = F.mse_loss(pred_audio, target_audio, reduction='none')

        spec_loss = spec_loss.view(N, -1).mean(dim=-1)
        spec_loss = spec_loss.view(-1, self.n_view - 1).mean(dim=-1)
        spec_loss = spec_weight * spec_loss 
        output['Spec Loss'] = spec_loss
        loss = spec_loss * self.generative_loss_ratio
        output['Loss'] = loss
        if evaluate:
            return output
        return loss


    def inference(self, inputs, pred_audio):
        # import pdb; pdb.set_trace()
        gt_audio = [inputs[f'audio_{i+1}'].unsqueeze(1) for i in range(1, self.n_view)]
        gt_audio = torch.cat(gt_audio, dim=1)
        audio_shape = gt_audio.shape 
        gt_audio = gt_audio.contiguous().view(-1, *audio_shape[2:])
        c = int(gt_audio.shape[1] // 2)

        if self.mono2binaural:
            audio_mix = gt_audio[:, :c, :] + gt_audio[:, c:, :]
            audio_input = self.wave2spec(audio_mix, return_complex=True).squeeze().detach()
            pred_audio = pred_audio.permute(0, 2, 3, 1)
            pred_audio = torch.view_as_complex(pred_audio.contiguous())
            pred_audio = torch.cat([pred_audio, audio_input[:, -1:, ...]], dim=1)
        
            pred_audio = torch.istft(
                input=pred_audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length
            ).unsqueeze(1)
            pred_left = (audio_mix + pred_audio) / 2
            pred_right = (audio_mix - pred_audio) / 2
            pred_audio = torch.cat([pred_left, pred_right], dim=1)
        else:
            raise NotImplementedError
        
        return {
            'pred_wave': pred_audio,
            'gt_wave': gt_audio
        }
    

    def encode_conditional_feature(self, inputs, cond_audio, augment):
        # import pdb; pdb.set_trace()
        # We always set the Img 1 as conditional view
        B = cond_audio.shape[0]

        # ------  Encode the conditional audio at the source viewpoint  --------- #
        if self.no_cond_audio:
            cond_audio_feat = None
        else:
            cond_audio_feat = self.audio_net(cond_audio, augment=augment)
            cond_audio_feat = torch.cat([cond_audio_feat.unsqueeze(1)] * (self.n_view - 1), dim=1)
            cond_audio_feat = cond_audio_feat.contiguous().view(-1, *cond_audio_feat.shape[2:])

        # ------  Encode the relative camera pose between different view  --------- #
        if self.no_vision:
            im_features = None
        else:
            single_im_features = []
            for i in range(0, self.n_view):
                im_feature = self.vision_net.forward_backbone(inputs[f'img_{i+1}'], augment=augment)
                single_im_features.append(im_feature)

            im_features = []
            for i in range(1, self.n_view):
                corr_feature = self.vision_net.forward_correlation(single_im_features[i], single_im_features[0])
                im_features.append(corr_feature.unsqueeze(1))

            im_features = torch.cat(im_features, dim=1)
            im_features = im_features.contiguous().view(-1, *im_features.shape[2:])

            # if self.use_gt_rotation:
            #     theta = torch.cat([inputs[f'relative_camera{i}_angle'].unsqueeze(1) for i in range(1, self.n_view)], dim=1)
            #     theta = theta.contiguous().view(theta.shape[0] * theta.shape[1], -1)
            #     theta = theta / 180.0 * np.pi
            #     im_features = self.rota_embedding(theta.float()).detach()

        # ------  Concat the conditional features  --------- #
        if self.no_vision and not self.no_cond_audio:
            cond_feats = cond_audio_feat
        elif not self.no_vision and self.no_cond_audio:
            cond_feats = im_features
        elif not self.no_vision and not self.no_cond_audio:
            cond_feats = torch.cat([cond_audio_feat, im_features], dim=-1)
        else:
            cond_feats = None

        return cond_feats
    
    def generate_audio_pair(self, inputs):
        # import pdb; pdb.set_trace()
        target_view_audio = [inputs[f'audio_{i+1}'].unsqueeze(1) for i in range(1, self.n_view)]

        target_view_audio = torch.cat(target_view_audio, dim=1)
        audio_shape = target_view_audio.shape 
        target_view_audio = target_view_audio.contiguous().view(-1, *audio_shape[2:])

        c = int(target_view_audio.shape[1] // 2)
        if self.mono2binaural:
            audio_mix = (target_view_audio[:, :c, :] + target_view_audio[:, c:, :])
            audio_input = self.wave2spec(audio_mix, return_real_imag=True).detach()
            audio_diff = (target_view_audio[:, :c, :] - target_view_audio[:, c:, :])
            audio_output = self.wave2spec(audio_diff, return_real_imag=True).detach()
        else:
            audio_input = self.wave2spec(target_view_audio[:, :c, :], return_mag_phase=True).detach()
            audio_output = self.wave2spec(target_view_audio[:, c:, :], return_mag_phase=True).detach()
        cond_audio = inputs['audio_1']
        return cond_audio, audio_input, audio_output


    def wave2spec(self, wave, return_complex=False, return_real_imag=False, return_mag_phase=False):
        # import pdb; pdb.set_trace()
        N, C, _ = wave.shape
        wave = wave.view(N * C, -1)
        spec = torch.stft(
            input=wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )
        spec = spec.contiguous().view(N, C, *spec.shape[1:])
        if return_complex:
            return spec
        elif return_real_imag:
            spec = torch.view_as_real(spec)
            spec = spec.permute(0, 1, 4, 2, 3)
            spec = spec.view(N, -1, *spec.shape[3:])
        elif return_mag_phase:
            mag, phase = spec.abs().unsqueeze(-1), spec.angle().unsqueeze(-1)
            mag = self.normalize_magnitude(mag)
            phase = self.normalize_phase(phase)
            spec = torch.cat([mag, phase], dim=-1)
            spec = spec.permute(0, 1, 4, 2, 3)
            spec = spec.contiguous().view(N, -1, *spec.shape[3:])
        else:
            # return log magnitude
            spec = spec.abs()
            spec = self.normalize_magnitude(spec)
        # spec: (N, C, F-1, T)
        spec = spec[:, :, :-1, :]
        return spec


    def normalize_magnitude(self, spec, inverse=False):
        # import pdb; pdb.set_trace()
        spec_min = -100
        spec_max = 60
        if not inverse:
            spec = torch.maximum(spec, torch.tensor(self.log_offset))
            spec = 20 * torch.log10(spec)
            spec = (spec - spec_min) / (spec_max - spec_min) * 2 - 1
            spec = torch.clip(spec, -1.0, 1.0)
            # spec = torch.log(spec + self.pr.log_offset)
        else:
            spec = (spec + 1) / 2
            spec = spec * (spec_max - spec_min) + spec_min
            spec = 10 ** (spec / 20)
        return spec


    def normalize_phase(self, phase, inverse=False):
        pi = 3.1416
        if not inverse:
            phase = phase / pi
            phase = torch.clip(phase, -1.0, 1.0)
        else:
            phase = phase * pi
        return phase


    # def freeze_param(self):
    #     if self.freeze_camera:
    #         for param in self.vision_net.parameters():
    #             param.requires_grad = False
    #     if self.freeze_audio:
    #         for param in self.audio_net.parameters():
    #             param.requires_grad = False
    #     if self.freeze_generative:
    #         for param in self.generative_net.parameters():
    #             param.requires_grad = False


    def score_model_performance(self, res):
        score = 1 / res['Loss']
        return score

    


def setup_reloc3r_relpose_model(model_args, device):
    if '224' in model_args:
        ckpt_path = 'siyan824/reloc3r-224'
    elif '512' in model_args:
        ckpt_path = 'siyan824/reloc3r-512'
        
    #reloc3r_relpose = Reloc3rRelpose.from_pretrained(ckpt_path)

    # Vision only
    #reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=False, has_audio_embedding=False)  # Reloc3rRelpose(img_size=256, has_audio=False) # Reloc3rRelpose(img_size=(512, 64), has_audio=False) # Reloc3rRelpose(img_size=512)
    # Audio only
    
    #reloc3r_relpose = Reloc3rRelpose(img_size=(512, 16), has_audio=True, has_audio_embedding=True) #Reloc3rRelpose(img_size=256)  #Reloc3rRelpose(img_size=(512, 64), is_audio=True, has_audio_embedding=False) #Reloc3rRelpose(img_size=512)

    print(model_args, type(model_args))
    if "DOA" in model_args:
        if "AudioDOAModel" in model_args:
            reloc3r_relpose = AudioDOAModel(embed_dim=1024, model_type=50, n_degs=360)
            print('Loading newly trained audio DOA model')

            # run 500ms or 1000ms next
            ckpt = torch.load("checkpoints/_egoexo4d-audio_doa_1000ms_alt_/checkpoint-best.pth", map_location=device)  # Load the model checkpoint
            # ckpt = torch.load("checkpoints/_slfm_hm3d-audio_doa_norm_/checkpoint-best.pth", map_location=device)  # Load the model checkpoint
        else:
            reloc3r_relpose = DOACameraPoseModel(input_embed_dim=1024+360, num_layers=20) #input_embed_dim = 360
            #ckpt = torch.load("checkpoints/_slfm_hm3d_doa_cp_/checkpoint-best.pth", map_location=device)  # Load the model checkpoint
            ckpt = torch.load("checkpoints/_egoexo4d-1000ms_doa_cp_slfm_nocorruption_/checkpoint-best.pth", map_location=device)  # Load the model checkpoint 
            
            # ckpt = torch.load("checkpoints/_slfm_hm3d_doa_cp_180/checkpoint-best.pth", map_location=device)  # Load the model checkpoint
            #ckpt = torch.load("checkpoints/_egoexo4d-500ms_doa_cp_/checkpoint-last.pth", map_location=device)  # Load the model checkpoint
    elif "policy" in model_args.lower():
        reloc3r_relpose = PolicyClassificationModel(model_type=50)
        print('Loading newly trained policy classification model')
        #  ckpt = torch.load("checkpoints/_egoexo4d-policy_doa_cp_1000ms_/checkpoint-50.pth", map_location=device)  # Load the model checkpoint
        
        ckpt = torch.load("checkpoints/_egoexo4d-policy_oursvsdoaandslfm_60ms_0-0_errs_audioweighted10x_200epochs_/checkpoint-best.pth", map_location=device)  # Load the model checkpoint
        # ckpt = torch.load("checkpoints/_egoexo4d-policy_oursvsdoa_60ms_/checkpoint-last.pth", map_location=device)  # Load the model checkpoint
        # ckpt = torch.load("checkpoints/_egoexo4d-policy_60ms_bce_/checkpoint-last.pth", map_location=device)   # _weight
    elif "slfm" in model_args.lower():
        reloc3r_relpose = SLfMNet()
        ckpt = torch.load("checkpoints/_egoexo4d-slfm_/checkpoint-last.pth", map_location=device)  # Load the model checkpoint
    else:
        
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True)
        # # # print('Loading newly trained model')

        # # # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_doa_embed_60ms-512_/checkpoint-last.pth", map_location=device)  # Load the model checkpoint # just ran 60ms last
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_embed_1000ms-512_/checkpoint-last.pth", map_location=device)  # Load the model checkpoint # just ran 60ms last
        
        # audio_only 60ms
        # reloc3r_relpose = Reloc3rRelpose(img_size=(512, 16), is_audio=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-audio_only_60ms_imgnorm-512_/checkpoint-best.pth", map_location=device)

        # audio_only 500ms
        # reloc3r_relpose = Reloc3rRelpose(img_size=(512, 48), is_audio=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-audio_only_500ms-512_/checkpoint-best.pth", map_location=device)

        # reloc3r_relpose = Reloc3rRelpose(img_size=(512, 96), is_audio=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-audio_only_1000ms-512_/checkpoint-best.pth", map_location=device)

        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_doa_embed_1000ms-512_/checkpoint-80.pth", map_location=device)  # Load the model checkpoint # just ran 60ms last

        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_doa_embed_1000ms-512_/checkpoint-last.pth")

        # Reloc3r + SLFM embedding
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-slfm_embed-512_/checkpoint-best.pth", map_location=device)
       
        #  Reloc3r naive audio + vision
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_embed_1000ms-512_/checkpoint-last.pth")

        # Reloc3r vision + SLfM Embedding
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-slfm_embed-512_/checkpoint-best.pth", map_location=device)

        # # Reloc3r vision + DOA 360 + embed
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_doa_and_embed=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_doa_360_and_embed_1000ms-512_/checkpoint-best.pth", map_location=device)

        # # # Reloc3r vision + DOA 360
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=False, use_doa=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_doa_360_1000ms-512_/checkpoint-last.pth", map_location=device)
        
        # # Reloc3r vision + audio cross attention
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_detr_cross_attention=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_cross_att_1000ms-512_/checkpoint-best.pth", map_location=device)
        
        # Reloc3r vision + SLfM + DOA 360 (OURS) -- the method released in this repo
        reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_doa_and_embed=True)
        ckpt = torch.load("checkpoints/_egoexo4d-vision_slfm_embed_and_doa_360_1000ms-512_/checkpoint-last.pth", map_location=device)

        # # Reloc3r vision + IMU
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=False, has_audio_embedding=False, has_imu=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_imu-512_/checkpoint-last.pth", map_location=device)

        # # Reloc3r vision + SLfM + DOA 360 + imu (OURS + imu)
        # reloc3r_relpose = Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, has_imu=True)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_slfm_doa_imu-512_/checkpoint-best.pth", map_location=device)

        # # SLFM Vision only
        # reloc3r_relpose = Reloc3rRelpose(img_size=(320, 240), has_audio=False, is_slfm=True, has_audio_embedding=False)
        # ckpt = torch.load("checkpoints/_slfm_hm3d-vision_only-512_/checkpoint-best.pth", map_location=device)

        # SLFM Audio only
        # reloc3r_relpose = Reloc3rRelpose(img_size=(256, 256), is_audio=True, is_slfm=True)
        # ckpt = torch.load("checkpoints/_slfm_hm3d-audio_only_norm-512_/checkpoint-best.pth", map_location=device)
        
        # SLFM Vision + Audio Embed
        # reloc3r_relpose = Reloc3rRelpose(img_size=(320, 240), has_audio=True, has_audio_embedding=True, is_slfm=True)
        # ckpt = torch.load("checkpoints/_slfm_hm3d-vision_audio_specnorm_embed-512_/checkpoint-last.pth", map_location=device)

        # SLFM Vision + Audio DOA Embed
        # reloc3r_relpose = Reloc3rRelpose(img_size=(320, 240), has_audio=True, has_audio_embedding=True, is_slfm=True)
        # ckpt = torch.load("checkpoints/_slfm_hm3d-vision_doa_embed-512_/checkpoint-best.pth", map_location=device)

        # EgoExo4d vision only
        # reloc3r_relpose = Reloc3rRelpose(img_size=(256, 256), has_audio=False, has_audio_embedding=False)
        # ckpt = torch.load("checkpoints/_egoexo4d-vision_only_FIXED-512_/checkpoint-best.pth", map_location=device)

        # reloc3r_relpose = Reloc3rRelpose(img_size=256)
        # ckpt = torch.load("checkpoints/_egoexo4d-corrupted_vision_test-512_/checkpoint-best.pth", map_location=device)
        

    #reloc3r_relpose = AudioDOAModel(embed_dim=1024, model_type=50) #AudioDOAModel()
    # ckpt = torch.load("checkpoints/_egoexo4d-gradient_blending-512_/checkpoint-best.pth", map_location=device)

    #ckpt = torch.load("checkpoints/_egoexo4d-doa_1src_2d_spectral_resnet50-512_/checkpoint-final.pth", map_location=device) # just ran final
    #  ckpt = torch.load("checkpoints/_egoexo4d-audio_doa_azimuth_1src_resnet50/checkpoint-best.pth", map_location=device) # 

    #  ckpt = torch.load("checkpoints/_egoexo4d-audio_only_1000ms-512_/checkpoint-best.pth", map_location=device)  # Load the model checkpoint
    # ckpt = torch.load("checkpoints/_egoexo4d-vision_only-512_/checkpoint-best.pth", map_location=device)  # Load the model checkpoint
    # ckpt = torch.load("checkpoints/_egoexo4d-vision_audio_embed_60ms-512_/checkpoint-final.pth", map_location=device)  # Load the model checkpoint

    if 'model' in ckpt: 
        ckpt = ckpt['model']
    new_ckpt = dict(ckpt)
    if any(k.startswith('dec_blocks2') for k in ckpt):
        for key, value in ckpt.items():
            if key.startswith('dec_blocks2'):
                new_ckpt[key.replace('dec_blocks2', 'dec_blocks')] = value
    ckpt = new_ckpt
    print(reloc3r_relpose.load_state_dict(ckpt, strict=False))
    
    # else:
    #     reloc3r_relpose = Reloc3rRelpose.from_pretrained(ckpt_path)

    reloc3r_relpose.to(device)
    reloc3r_relpose.eval()
    #print('Model loaded from ', ckpt_path)
    
    return reloc3r_relpose


@torch.no_grad()
def inference_relpose(batch, model, device, use_amp=False): 
    # to device. 
    for view in batch:
        for name in 'img camera_intrinsics camera_pose audio_spec doas audio_embedding input_doa yamnet_embed policy_decision waveform'.split() + ["vision_error", "audio_error", "vision_rot_error", "audio_rot_error", "vision_trans_error", "audio_trans_error"]:  
            if name not in view:
                continue

            view[name] = view[name].float().to(device, non_blocking=True)
    # forward. 
    view1, view2 = batch

    if "doas" in view1:
        
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            
            doa, emb = model(view1, return_embedding=True)  # audio model
            if "slfm" in view1["dataset"][0].lower():
                emb_dir = f"./embeddings_slfm_doa_180/"
                for i in range(emb.shape[0]):
                    emb_i = emb[i:i+1].cpu().numpy()
                    label = view1['label'][i]
                    take_name = '-'.join(label.split('-')[:2])
                    os.makedirs(os.path.join(emb_dir, take_name), exist_ok=True)
                    torch.save(emb_i, os.path.join(emb_dir, take_name, f'embedding_{label}.pt'))
            else:  
                emb_dir = f"./embeddings_doa_{view1['sound_size'][0].item()}ms_alt/"

                for i in range(emb.shape[0]):
                    emb_i = emb[i:i+1].cpu().numpy()
                    label = view1['label'][i]
                    take_name = view1['take_name'][i]
                    os.makedirs(os.path.join(emb_dir, take_name), exist_ok=True)
                    doa_i = doa[i:i+1].cpu().numpy()

                    torch.save(emb_i, os.path.join(emb_dir, take_name, f'embedding_{label}.pt'))
                    torch.save(doa_i, os.path.join(emb_dir, take_name, f'doa_{label}.pt'))
        
        return doa
    elif "waveform" in view1:
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            inputs = {'img_1': view1['img'], 
                      'img_2': view2['img'],
                      'audio_1': view1['waveform'],
                      'audio_2': view2['waveform'],}
            
            output, features = model(inputs, evaluate=True)
            emb_dir = "./embeddings_slfm_egoexo4d/"
            
            for i in range(features.shape[0]):
                feat_1 = features[i:i+1].cpu().numpy()
                feat_2 = features[i:i+1].cpu().numpy()
                label1 = view1['label'][i]
                label2 = view2['label'][i]

                take_name = view1['take_name'][i]
                os.makedirs(os.path.join(emb_dir, take_name), exist_ok=True)
                torch.save(feat_1, os.path.join(emb_dir, take_name, f'embedding1_{label1}.pt'))
                torch.save(feat_2, os.path.join(emb_dir, take_name, f'embedding2_{label2}.pt'))
        return output

    elif "policy_decision" in view1:
        # try:
        #     macs = profile_macs(model, view1) # per sample
        #     print(f"MACs: {((macs/view1['save_embedding'].shape[0]) /1e9):.3f} GMACs")
        #     raise Exception("done")
        # except Exception as e:
        #     print(f"MACs profiling failed: {e}")
        #     raise e
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            # torch.cuda.reset_peak_memory_stats()
            # torch.cuda.synchronize()
            # start_mem = torch.cuda.memory_allocated()
            logits = model(view1, view2)  # policy classification model
            pred = (torch.sigmoid(logits) > 0.5).int()
            # torch.cuda.synchronize()
            # end_mem = torch.cuda.memory_allocated()
            # peak_mem = torch.cuda.max_memory_allocated()
            # raise Exception("Memory transferred", (peak_mem - start_mem)/len(view1['label'])/1e6, "MB")
            return pred
        
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        _, pose2 = model(view1, view2) # joint default
            
        # try:
        #     macs = profile_macs(model, args=(view1, view2)) # per sample
        #     print(f"MACs: {((macs/view1['save_embedding'].shape[0]) /1e9):.3f} GMACs")
        #     raise Exception("done")
        # except Exception as e:
        #     print(f"MACs profiling failed: {e}")
        #     raise e
        track_memory = False
        if "save_embedding" in view1 and view1["save_embedding"][0]:
            if track_memory:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()
                _, pose2, embedding1, embedding2 = model(view1, view2, return_embedding=True)
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()
                #raise Exception("Memory transferred", (peak_mem - start_mem)/len(view1['label'])/1e6, "MB")
            else:
                _, pose2, embedding1, embedding2 = model(view1, view2, return_embedding=True)
                
            sound_size = view1["sound_size"] 
            if "slfm" in view1["dataset"][0].lower():
                if "doas" in view1:
                    emb_dir = f"./embeddings_slfm_doa_180/"
                else:
                    emb_dir = f"./embeddings_slfm_audio_norm/"
            else:
                emb_dir = f"./embeddings_{sound_size[0].item()}ms/" # batch[0].get('embedding_dir', './embeddings/')

            # Save each embedding in the batch separately
            for i in range(embedding1.shape[0]):
                emb1 = embedding1[i:i+1].cpu().numpy()
                emb2 = embedding2[i:i+1].cpu().numpy()
                
                label1 = view1['label'][i]
                label2 = view2['label'][i]

                if "slfm" in view1["dataset"][0].lower():
                    take_name = '-'.join(label1.split('-')[:2])
                    # label1 = view1['label'][i]
                    # label2 = view2['label'][i]
                    # os.makedirs(os.path.join(emb_dir, take_name), exist_ok=True)

                    # torch.save(emb1, os.path.join(emb_dir, take_name, f'embedding1_{label1}.pt'))
                    # torch.save(emb2, os.path.join(emb_dir, take_name, f'embedding2_{label2}.pt'))
                else:
                    take_name = view1['take_name'][i]

                os.makedirs(os.path.join(emb_dir, take_name), exist_ok=True)

                torch.save(emb1, os.path.join(emb_dir, take_name, f'embedding1_{label1}.pt'))
                torch.save(emb2, os.path.join(emb_dir, take_name, f'embedding2_{label2}.pt'))
        else:
            if track_memory:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()
                _, pose2 = model(view1, view2)
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()
                print("Memory transferred", (peak_mem - start_mem)/len(view1['label'])/1e6, "MB")
                raise Exception(f"Memory used for batch: {(end_mem - start_mem)/1e6:.3f} MB, Peak memory during: {peak_mem/1e6:.3f} MB")
            else:
                _, pose2 = model(view1, view2)                
        
    pose2to1 = pose2["pose"]
    # method = "vision_only"
    for i in range(pose2to1.shape[0]):
        pose_i = pose2to1[i:i+1].cpu().numpy()
        label1 = view1['label'][i]
        label2 = view2['label'][i]

            # source_frame_id= view1["source_frame_id"][i]
            # target_frame_id = view2["target_frame_id"][i]

            # take_name = view1['take_name'][i]
        # os.makedirs(os.path.join("./result_poses/", method, take_name), exist_ok=True)
        # np.save(os.path.join("./result_poses/", method, take_name, f'predicted_{source_frame_id}_to_{target_frame_id}.npy'), pose_i)
    return pose2to1

# if __name__ == "__main__":
#     import pdb; pdb.set_trace()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     gpus = torch.cuda.device_count()
#     gpu_ids = list(range(gpus))
#     net = AudioCondUNet().to(device)
#     net = nn.DataParallel(net, device_ids=gpu_ids)
#     spec_input = torch.rand(16, 2, 256, 256).to(device)
#     visual_input = torch.rand(16, 256).to(device)
#     out = net(spec_input, visual_input)
