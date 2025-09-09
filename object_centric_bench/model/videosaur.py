from einops import rearrange
import torch as pt
import torch.nn as nn

from .dinosaur import DINOSAURT


class VideoSAUR(DINOSAURT):
    """
    Zadaianchuk et al. Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities. NeurIPS 2023.

    Different from DINOSAURT in the decoder, i.e., SlotMixerDecoder, and the loss, i.e., extra FeatureSimilarityLoss.
    """

    # segment acc: attent > attent2


class SlotMixerDecoder(nn.Module):
    """http://arxiv.org/abs/2206.06922"""

    def __init__(self, embed_dim, posit_embed, allocat, attent, render):
        super().__init__()
        self.posit_embed = posit_embed  # 1d
        self.norm_m = nn.LayerNorm(embed_dim, eps=1e-5)

        assert isinstance(allocat, nn.TransformerDecoder)
        for tfdb in allocat.layers:
            assert isinstance(tfdb, nn.TransformerDecoderLayer)
            tfdb.self_attn = nn.Identity()
            tfdb.self_attn.batch_first = True  # for compatiblity
            tfdb.dropout1 = nn.Identity()
            tfdb._sa_block = lambda *a, **k: a[0]
        self.allocat = allocat  # Tfd

        self.norm_q = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm_k = nn.LayerNorm(embed_dim, eps=1e-5)

        assert isinstance(attent, nn.MultiheadAttention)
        if attent._qkv_same_embed_dim:
            chunks = attent.in_proj_weight.chunk(3, 0)
            attent.q_proj_weight = nn.Parameter(chunks[0])
            attent.k_proj_weight = nn.Parameter(chunks[1])
            attent.in_proj_weight = None
            attent._qkv_same_embed_dim = False
        del attent.v_proj_weight, attent.out_proj.weight
        attent.register_buffer(
            "v_proj_weight", pt.eye(attent.embed_dim, dtype=pt.float)
        )
        attent.out_proj.register_buffer(
            "weight", pt.eye(attent.out_proj.in_features, dtype=pt.float)
        )
        self.attent = attent  # mha

        self.render = render  # MLP

    def forward(self, input, slotz):
        """
        input: destructed target, height and width
        slotz: slots, shape=(b,n,c)
        """
        h, w = input
        b, n, c = slotz.shape
        x = pt.zeros([b, h * w, c], dtype=slotz.dtype, device=slotz.device)

        query, pe = self.posit_embed(x, True)  # bmc
        memory = self.norm_m(slotz)  # bnc
        query = self.allocat(query, memory=memory)  # bmc

        q = self.norm_q(query)  # bmc
        k = self.norm_k(slotz)  # bnc
        v = slotz  # bnc
        slotmix, attent = self.attent(q, k, v)  # bmc bmn
        slotmix = slotmix + pe  # bmc
        recon = self.render(slotmix)  # bmc

        attent = attent.permute(0, 2, 1)  # bnm  # to match outside rearrange
        return recon, attent
