from diffusers import DDPMScheduler, UNet2DConditionModel
from einops import rearrange
import torch as pt
import torch.nn as nn


class SlotDiffusion(nn.Module):
    """SlotDiffusion model for images."""

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        mediat,  # VQVAE originally
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.mediat = mediat
        self.decode = decode
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):
        """
        - input: image in shape (b,c,h,w)
        - condit: condition in shape (b,n,c)
        """
        feature = self.encode_backbone(input)  # (b,c,h,w)
        b, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        query = self.initializ(b if condit is None else condit)  # (b,n,c)
        slotz, attent = self.aggregat(encode, query)
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)

        with pt.inference_mode(True):
            encode1, zidx, quant, decode1 = self.mediat(input)
        b, c, h, w = quant.shape
        quant = quant.clone()  # (b,c,h,w)

        clue = quant
        recon, noise = self.decode(clue, slotz)

        return slotz, attent, noise, recon


class ConditionDiffusionDecoder(nn.Module):
    """SlotDiffusion's decoder."""

    def __init__(self, noise_sched, backbone):
        super().__init__()
        self.noise_sched = noise_sched
        self.backbone = backbone

    def forward(self, input, slotz):
        """
        - input: target to be destructed, shape=(b,c,h,w)
        - slotz: slots, shape=(b,n,c)
        """
        noise, timestep, noisy = self.noise_sched(input)
        decode = self.backbone(noisy, timestep, slotz)
        return decode, noise


class NoiseSchedule(nn.Module):

    def __init__(
        self,
        beta_schedule="linear",
        beta_start=0.00085,
        beta_end=0.012,
        num_train_timesteps=1000,
        prediction_type="epsilon",
    ):
        super().__init__()
        kwargs = {
            # "_class_name": "DDPMScheduler",
            # "_diffusers_version": "0.21.4",
            # "beta_end": 0.012,
            # "beta_schedule": "linear",
            # "beta_start": 0.00085,
            "clip_sample": False,
            "clip_sample_range": 1.0,
            "dynamic_thresholding_ratio": 0.995,
            # "num_train_timesteps": 1000,
            # "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "steps_offset": 1,
            "thresholding": False,
            "timestep_spacing": "leading",
            "trained_betas": None,
            "variance_type": "fixed_small",
        }
        self.noise_scheduler = DDPMScheduler(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            **kwargs
        )

    @pt.no_grad()
    def forward(self, input):
        b = input.size(0)
        timesteps = pt.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b,),
            device=input.device,
        )
        timesteps = timesteps.long()
        noise = pt.randn_like(input)
        noisy = self.noise_scheduler.add_noise(input, noise, timesteps)
        return noise, timesteps, noisy


class UNet2dCondition(UNet2DConditionModel):

    def forward(self, input, timestep, context):
        return super().forward(input, timestep, context).sample
