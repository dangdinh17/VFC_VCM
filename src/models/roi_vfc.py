import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from typing import List, Optional, Union, Dict

from .backbone import FeatureExtraction, PerceptionExtraction
from .motion import MotionEstimation, MotionCompensation, MotionEncoder, MotionDecoder, HyperiorEntropyEncoder
from .roi_conditional import ROI_Guided_Conditional_Encoder, ROI_Guided_Conditional_Decoder
from .buffer import FeatureBuffer
from ..entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from ..utils.stream_helper import get_downsampled_shape, encode_fea, decoder_fea, filesize

from collections import OrderedDict
import time
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ROI_VFC(nn.Module):
    """A minimal TransVFC-like pipeline combining all components."""

    def __init__(self, img_channels=3, feat_channels=64, y_chan=96, lambda_rd=0.01):
        super().__init__()
        self.feat_extraction = FeatureExtraction(pretrained=False)
        self.perception_extraction = PerceptionExtraction(backbone_name='resnet50', pretrained=False)
        
        self.motion_estimation = MotionEstimation(channels=feat_channels)
        self.motion_encoder = MotionEncoder(in_channels=feat_channels, out_channels=feat_channels)
        self.motion_decoder = MotionDecoder(in_channels=feat_channels, out_channels=feat_channels)
        self.motion_compensation = MotionCompensation(channels=feat_channels)
        
        self.channel_reduction = nn.Conv2d(256, feat_channels, kernel_size=1)
        self.channel_restoration = nn.Conv2d(feat_channels, 256, kernel_size=1)

        self.roi_guided_conditional_encoder = ROI_Guided_Conditional_Encoder(feat_chan=feat_channels, out_chan=y_chan)
        self.roi_guided_conditional_decoder = ROI_Guided_Conditional_Decoder(feat_chan=feat_channels, out_chan=y_chan)

        self.hyperior_entropy_parameter = HyperiorEntropyEncoder(in_channels=feat_channels, out_channels=2*y_chan)
    
        # entropy modules (hyperprior + gaussian conditional), similar to TIC
        # self.entropy_bottleneck = EntropyBottleneck(feat_channels)
        # self.gaussian_conditional = GaussianConditional(None)

        self.bit_estimator = BitEstimator(channel=feat_channels)
        self.gaussian_encoder = GaussianEncoder()
        
        self.feature_buffer = FeatureBuffer(max_size=3)
        self.lambda_rd = lambda_rd
        
    def reset_buffer(self):
        self.feature_buffer.reset()
    
    def get_ref(self, F_cur):
        ref = self.feature_buffer.get()
        if ref is None:
            ref = F_cur
        return ref
    def update_buffer(self, F_cur_hat):
        self.feature_buffer.push(F_cur_hat.detach().clone())

    def extract_feature(self, x):
        with torch.no_grad():
            fea = self.feat_extraction(x)
        return fea
    
    def quantize(self, x):
        return torch.round(x)
    
    @staticmethod
    def pyramid_reduction(pyramid_feats: Union[List[torch.Tensor], Dict[str, torch.Tensor], tuple] = None, channel_reduction = None):
        pyramid_feats["p2"] = channel_reduction(pyramid_feats["p2"])
        pyramid_feats["p3"] = channel_reduction(pyramid_feats["p3"])
        pyramid_feats["p4"] = channel_reduction(pyramid_feats["p4"])
        pyramid_feats["p5"] = channel_reduction(pyramid_feats["p5"])
        pyramid_feats["p6"] = channel_reduction(pyramid_feats["p6"])
        return pyramid_feats
    
    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.bit_estimator.update(force=force, entropy_coder=self.entropy_coder)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    @staticmethod
    def get_y_bits_probs(y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    @staticmethod
    def get_z_bits_probs(z, bit_estimator):
        prob = bit_estimator(z + 0.5) - bit_estimator(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob
    
    
    def get_downsampled_shape(height, width, p):

        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def mse_pyr_feat_loss(self, pyr1, pyr2):
        """Compute MSE loss between two pyramid dicts."""
        mse_loss = 0.0
        for key in pyr1.keys():
            if key in pyr2:
                mse_loss += F.mse_loss(pyr1[key], pyr2[key])
        return mse_loss
    
    def forward(self, F_cur):
        # 1. Feature extraction
        F_ref = self.get_ref(F_cur)
        f_cur = self.channel_reduction(F_cur)
        f_ref = self.channel_reduction(F_ref)
        # 2. Motion estimation
        m_t = self.motion_estimation(f_cur, f_ref)
        z = self.motion_encoder(m_t)

        z_hat = self.quantize(z)
        # z_probs = self.bit_estimator(z_hat + 0.5) - self.bit_estimator(z_hat - 0.5)

        m_hat = self.motion_decoder(z_hat)
        f_cur_tilde = self.motion_compensation(f_ref, m_hat)

        F_cur_tilde = self.channel_restoration(f_cur_tilde)

        with torch.no_grad():
            Enc_pyr = self.perception_extraction(F_cur)
            Dec_pyr = self.perception_extraction(F_cur_tilde)

        enc_pyr = self.pyramid_reduction(Enc_pyr, self.channel_reduction)
        dec_pyr = self.pyramid_reduction(Dec_pyr, self.channel_reduction)

        scales_hat, means_hat = self.hyperior_entropy_parameter(f_cur_tilde).chunk(2, 1)
        
        y = self.roi_guided_conditional_encoder(f_cur, f_cur_tilde, enc_pyr)
        
        y_res = y - means_hat
        y_q = self.quantize(y_res)
        y_hat = y_q + means_hat

        f_cur_hat = self.roi_guided_conditional_decoder(y_hat, f_cur_tilde, dec_pyr)

        # 4. Motion compensation
        F_cur_hat = self.channel_restoration(f_cur_hat)
        self.update_buffer(F_cur_hat)

        y_for_bit = y_q
        z_for_bit = z_hat
        
        total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
        total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator)
        
        mse_f = F.mse_loss(F_cur_hat, F_cur)
        mse_c = F.mse_loss(F_cur_tilde, F_cur)
        mse_p = self.mse_pyr_feat_loss(enc_pyr, dec_pyr)
        
        fea_shape = F_cur.size()
        pixels = fea_shape[0] * fea_shape[2] * fea_shape[3]
        
        bpp_y = total_bits_y / pixels if pixels > 0 else 0.0
        bpp_z = total_bits_z / pixels if pixels > 0 else 0.0    
        bpp = bpp_y + bpp_z
        
        return {
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp,
            "mse_f": mse_f,
            "mse_c": mse_c,
            "mse_p": mse_p,
            "z": z,
            "y": y,
            "bit_y": total_bits_y,
            "bit_z": total_bits_z,
            "bit": total_bits_y + total_bits_z,
            "y_hat": y_hat,
            "z_hat": z_hat,
            "F_cur_hat": F_cur_hat,
            "f_cur_hat": f_cur_hat,
            "F_ref": F_ref,
            "f_ref": f_ref,
        }

    def compress(self, z, y, f_ref):
        self.entropy_coder.reset_encoder()
        
        try:
            print("z.shape:", tuple(z.size()))
            z_dtype = z.dtype if hasattr(z, "dtype") else None
            z_min = float(z.min().item())
            z_max = float(z.max().item())
            z_mean = float(z.float().mean().item())
        except Exception:
            z_dtype = None
            z_min = z_max = z_mean = None
        print(f"z.dtype: {z_dtype}, z.min: {z_min}, z.max: {z_max}, z.mean: {z_mean}")

        z_hat = self.quantize(z)
        print("z_hat.shape:", tuple(z_hat.size()))
        try:
            z_min = float(z_hat.min().item())
            z_max = float(z_hat.max().item())
            z_mean = float(z_hat.float().mean().item())
        except Exception:
            z_min = z_max = z_mean = None
        print(f"z_hat.dtype: {z_hat.dtype}, z_hat.min: {z_min}, z_hat.max: {z_max}, z_hat.mean: {z_mean}")


        _ = self.bit_estimator.encode(z_hat)
        
        m_hat = self.motion_decoder(z_hat)
        f_cur_tilde = self.motion_compensation(f_ref, m_hat)  # we don't have ref feature here, but motion compensation should be designed to handle this case (e.g. return zeros)
        scales_hat, means_hat = self.hyperior_entropy_parameter(f_cur_tilde).chunk(2, 1)
        print("ads")
        y_res = y - means_hat
        y_q = self.quantize(y_res)  # ensure integer type for gaussian encoder
        _ = self.gaussian_encoder.encode(y_q, scales_hat)
        
        string = self.entropy_coder.flush_encoder()
        bit = len(string) * 8
    
        return {
            "string": string,
            "bit": bit,
            "z_size": z_hat.size()[-2:]
         } # we need the shape to decode z; y

    def decompress(self, F_ref, string, shape):
        self.entropy_coder.set_stream(string)
        device = next(self.parameters()).device
        f_ref = self.channel_reduction(F_ref)
        z_size = shape
        z_hat = self.bit_estimator.decode_stream(z_size).to(device)
        m_cur_hat = self.motion_decoder(z_hat)
        f_cur_tilde = self.motion_compensation(f_ref, m_cur_hat)
        F_cur_tilde = self.channel_restoration(f_cur_tilde)
        
        scales_hat, means_hat = self.hyperior_entropy_parameter(f_cur_tilde).chunk(2, 1)
        y_q = self.gaussian_encoder.decode_stream(scales_hat).to(device)
        y_hat = y_q + means_hat
        
        dec_pyr = self.perception_extraction(F_cur_tilde)
        dec_pyr = self.pyramid_reduction(dec_pyr, self.channel_reduction)
        f_cur_hat = self.roi_guided_conditional_decoder(y_hat, f_cur_tilde, dec_pyr)
        F_cur_hat = self.channel_restoration(f_cur_hat)
        
        F_cur_hat = F_cur_hat.clamp(0, 1)
        
        return {
            "F_cur_hat": F_cur_hat,
        }
        
    def encode_decode(self, F_cur, output_path=None):
        
        if output_path is not None:
            forward = self.forward(F_cur)
            z = forward['z']
            y = forward['y']
            F_ref = forward['F_ref']
            f_ref = forward['f_ref']
            encoded = self.compress(z, y, f_ref)
            encode_fea(encoded['string'], output_path)
            bits = filesize(output_path) * 8
            string = decoder_fea(output_path)
            start = time.time()
            decoded = self.decompress(F_ref, string, encoded['z_size']) 
            decoding_time = time.time() - start
            result = {
                # "F_hat": decoded["F_cur_hat"],
                "bit_y": 0,
                "bit_z": 0,
                "bit_mv_y": 0,
                "bit_mv_z": 0,
                "bit": bits,
                "forward_bit": forward['bit'].detach().item(),
                "decoding_time": decoding_time,
            }
            return result
        
        encoded = self.forward(F_cur)
        result = {
            # "F_hat": encoded['F_cur_hat'],
            "y": encoded['y'].shape,
            "bit_y": encoded['bit_y'].item(),
            "bit_z": encoded['bit_z'].item(),
            "bit": encoded['bit'].item(),
            "bpp": encoded['bpp'].item(),
            "decoding_time": 0
        }
        return result