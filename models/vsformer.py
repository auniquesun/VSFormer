from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from fairscale.nn import checkpoint_wrapper
import torch.nn as nn
import torchvision.models as models

import timm
from timm.models.layers import DropPath


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


class Sequential(nn.Sequential):
    def forward(self, *x):
        for module in self:
            if type(x) == tuple:
                x = module(*x)
            else:
                x = module(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
        vis_flag: bool = False,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of channels query and key input channels are projected to,
            for computing the attention matrix. Defaults to number `num_q_input_channels`
        :param num_v_channels: Number of channels value input channels are projected to.
            Defaults to `num_qk_channels`.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=False)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=False)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=False)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels)
        self.dropout = nn.Dropout(dropout)

        self.vis_flag = vis_flag

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param attn_mask: Boolean attention mask. Not needed/supported yet.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """
        if attn_mask is not None:
            raise NotImplementedError("attention masks not supported yet")

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        attn = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "b j -> (b h) () j", h=self.num_heads)
            attn_max_neg = -torch.finfo(attn.dtype).max
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn_scores = attn.softmax(dim=-1)
        # 这里 q,k 做完attention，还要做dropout，想想自己之前是否忽略了这一点
        attn = self.dropout(attn_scores)

        o = torch.einsum("b i j, b j c -> b i c", attn, v)
        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        if self.vis_flag:
            return self.o_proj(o), attn_scores
        else:
            return self.o_proj(o)


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
        vis_flag: bool = False,
    ):
        """Multi-head self-attention (see `MultiHeadAttention` and for details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
            vis_flag=vis_flag,
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        """Multi-head attention of input `x` to itself after applying layer normalization to the input."""
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        drop_path_rate: float = 0.0,
        atten_drop: float = 0.1,
        mlp_drop: float = 0.5,
        vis_flag: bool = False,
    ):
        super().__init__()

        self.sa = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=atten_drop,
            vis_flag=vis_flag)
        self.atten_drop = nn.Dropout(atten_drop)
        self.mlp = MLP(num_channels, widening_factor)
        self.mlp_drop = nn.Dropout(mlp_drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > .0 else nn.Identity()

        self.vis_flag = vis_flag

    def forward(self, img_embs):
        tmp = img_embs
        if self.vis_flag:
            x, attn_scores = self.sa(img_embs)
        else:
            x = self.sa(img_embs)
        x = self.atten_drop(x) + tmp
        x = self.drop_path(x)

        tmp = x
        x = self.mlp(x)
        x = self.mlp_drop(x) + tmp
        x = self.drop_path(x)

        if self.vis_flag:
            return x, attn_scores
        else:
            return x


class MLP(Sequential):
    def __init__(self, num_channels: int, widening_factor: int):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels),
        )
        # dropout 没有显式加在这里，而是加在了 Residual


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float, drop_path_rate: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > .0 else nn.Identity()

    def forward(self, *args, **kwargs):
        # x 是 SelfAttentionLayer/MLP 的输出
        x = self.module(*args, **kwargs)
        # 对 x 做dropout，再做残差连接
        return self.drop_path(self.dropout(x) + args[0])


class BaseImageClassifier(nn.Module):
    def __init__(self, model_name='resnet18', base_feature_dim=512, num_channels=512, num_classes=40, pretrained=True, weights=None):
        super().__init__()
        if not pretrained:
            weights = None

        # --- option 1: using torchvision model 
        # NOTE use pretrained weights by default
        if model_name == 'alexnet':
            model = models.alexnet(weights=weights)
        elif model_name == 'vgg11':
            # model = models.vgg11_bn(weights=weights)  # bad peformance
            model = models.vgg11(weights=weights)
        elif model_name == 'resnet18':
            model = models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            model = models.resnet34(weights=weights)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=weights)

        if 'alexnet' in model_name:
            modules = list(model.children())[:-1] + [
                nn.Sequential(  # adapt AlexNet `classifier` according to our needs
                    Rearrange('b c h w -> b (c h w)', c=256, h=6, w=6), # NOTE this conversion is necessary
                    nn.Dropout(0.5),
                    nn.Linear(in_features=9216, out_features=4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=4096, out_features=num_channels),
                    nn.ReLU(inplace=True),
                )
            ]
            self.feature_extractor = nn.Sequential(*modules)
            self.classifier = nn.Linear(num_channels, num_classes)
        elif 'vgg' in model_name:
            modules = list(model.children())[:-1] + [
                nn.Sequential(  # adapt VGG `classifier` according to our needs
                    Rearrange('b c h w -> b (c h w)', c=512, h=7, w=7), # NOTE this conversion is necessary
                    nn.Linear(in_features=25088, out_features=4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=4096, out_features=num_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                )
            ]
            self.feature_extractor = nn.Sequential(*modules)
            self.classifier = nn.Linear(num_channels, num_classes)
        elif 'resnet' in model_name:
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            if base_feature_dim != num_channels:
                self.classifier = nn.Sequential(
                    nn.Linear(base_feature_dim, num_channels),
                    nn.BatchNorm1d(num_channels),
                    nn.ReLU(),
                    nn.Linear(num_channels, num_classes))
            else:
                self.classifier = nn.Linear(num_channels, num_classes)
        # --- option 2: using timm model 
        # self.model = timm.create_model(model_name, pretrained=pretrained)
        # self.model.fc = nn.Linear(512, num_classes)

    def forward(self, imgs):
        batch = imgs.shape[0]
        img_feats = self.feature_extractor(imgs)
        # img_feats: [batch, num_channels]
        img_feats = img_feats.reshape(batch, -1)
        # img_classes: [batch, num_classes]
        img_classes = self.classifier(img_feats)
        return img_classes


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        max_dpr: float = 0.0,
        atten_drop: float = 0.0,
        mlp_drop: float = 0.0,
        activation_checkpointing: bool = False, 
        vis_flag: bool = False,
    ):
        super().__init__()

        dpr_list = [dpr.item() for dpr in torch.linspace(0, max_dpr, num_layers)]
        self.sa_layers = nn.ModuleList()
        for i in range(num_layers):
            self.sa_layers.append(SelfAttentionLayer(
                                    num_heads=num_heads,
                                    num_channels=num_channels,
                                    num_qk_channels=num_qk_channels,
                                    num_v_channels=num_v_channels,
                                    widening_factor=widening_factor,
                                    drop_path_rate=dpr_list[i],
                                    atten_drop=atten_drop,
                                    mlp_drop=mlp_drop,
                                    vis_flag=vis_flag))
        
        self.vis_flag = vis_flag

        if activation_checkpointing:
            self.sa_layers = [checkpoint_wrapper(layer) for layer in self.sa_layers]

    def forward(self, img_embs, pos_embs=None):
        '''
        Args:
            img_embs: [batch, num_views, dim_view_feats]
            pos_embs: [batch, num_views, dim_view_feats]
        Return:
            x: [batch, num_views, dim_view_feats]
        '''
        attn_scores_list = []
        x = img_embs
        for sa_layer in self.sa_layers:
            if self.vis_flag:
                x, attn_scores = sa_layer(x)
                attn_scores_list.append(attn_scores)
            else:
                x = sa_layer(x)

        if self.vis_flag:
            return x, attn_scores_list
        else:
            return x


class ClsHead(nn.Module):
    def __init__(
        self,
        num_layers: int, 
        num_channels: int,
        num_classes: int = 40, 
        use_cls_token: bool = False,
        use_max_pool: bool = False,
        use_mean_pool: bool = False,
    ):
        super().__init__()

        self.use_cls_token = use_cls_token
        self.use_max_pool = use_max_pool
        self.use_mean_pool = use_mean_pool

        if self.use_cls_token or self.use_max_pool or self.use_mean_pool:
            num_transition_channels = num_channels
        else:
            num_transition_channels = 2*num_channels
            
        # option 1. 3-layer MLP
        if num_layers == 3:
            self.decoder = nn.Sequential(
                nn.BatchNorm1d(num_transition_channels),
                nn.ReLU(),
                nn.Linear(num_transition_channels, num_channels),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(),
                nn.Linear(num_channels, num_channels//2),
                nn.BatchNorm1d(num_channels//2),
                nn.ReLU(),
                nn.Linear(num_channels//2, num_classes))
        # option 2. 2-layer MLP
        elif num_layers == 2:
            self.decoder = nn.Sequential(
                nn.BatchNorm1d(num_transition_channels),
                nn.ReLU(),
                nn.Linear(num_transition_channels, num_channels),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(),
                nn.Linear(num_channels, num_classes))
        # option 3. 1-layer Linear
        elif num_layers == 1:
            self.decoder = nn.Linear(num_transition_channels, num_classes)
        else:
            raise ValueError(f'`num_layers` = {num_layers}, it should be chosen from [1,2,3]')

    def forward(self, x):
        '''
            Args:
                x: [batch, num_latents, num_channels]

        '''
        if self.use_cls_token:
            # cls_feats: [batch, num_channels]
            #   the first latent is `cls_token`
            cls_feats = x[:, :1, :].squeeze()   # NOTE `squeeze()` should be applied here

        elif self.use_max_pool:
            # cls_feats: [batch, num_channels]
            cls_feats = torch.max(x, dim=1)[0]

        elif self.use_mean_pool:
            # cls_feats: [batch, num_channels]
            cls_feats = torch.mean(x, dim=1)

        else:   # concatenation of max & mean pooling
            # cls_feats: [batch_size, 2*num_channels]
            cls_feats = torch.cat([x.max(1)[0], x.mean(1)], dim=1)

        # output: [batch_size, num_classes]
        output = self.decoder(cls_feats)

        return output
        

class VSFormer(nn.Module):
    def __init__(self,
        base_feature_extractor: nn.Sequential,
        base_model_name: str = 'alexnet', 
        base_feature_dim: int = 512, 
        num_layers: int = 6,
        num_heads: int = 6,
        num_channels: int = 384,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        max_dpr: float = 0.0,
        atten_drop: float = 0.0,
        mlp_drop: float = 0.0,
        num_views: int = 20,
        clshead_layers: int = 2,
        num_classes: int = 40,
        use_cls_token: bool = False, 
        use_max_pool: bool = False,
        use_mean_pool: bool = False,
        use_pos_embs: bool = False,
        vis_flag: bool = False,
        activation_checkpointing: bool = False,):
        super().__init__()

        # use existing conv architecture (e.g. resnet18) for the input image feature extraction
        #   base_feature_extractor has removed last `fc` layer
        self.base_feature_extractor = base_feature_extractor

        self.base_feature_dim = base_feature_dim
        self.num_channels = num_channels
        self.num_views = num_views

        self.base_model_name = base_model_name
        if 'resnet' in self.base_model_name and self.base_feature_dim != self.num_channels:
            self.linear_proj = nn.Linear(self.base_feature_dim, self.num_channels)
            self.bn1d = nn.BatchNorm1d(self.num_channels)
            self.act = nn.ReLU()

        self.use_cls_token = use_cls_token
        self.use_pos_embs = use_pos_embs

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(self.num_channels))
            if self.use_pos_embs:
                self.pos_embs = nn.Parameter(torch.empty(1+self.num_views, self.num_channels))
        else:
            if self.use_pos_embs:
                self.pos_embs = nn.Parameter(torch.empty(self.num_views, self.num_channels))
        self._init_parameters()

        self.vis_flag = vis_flag

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            widening_factor=widening_factor,
            max_dpr=max_dpr,
            atten_drop=atten_drop,
            mlp_drop=mlp_drop,
            activation_checkpointing=activation_checkpointing,
            vis_flag=vis_flag)

        self.cls_head = ClsHead(
            num_layers=clshead_layers,
            num_channels=num_channels,
            num_classes=num_classes,
            use_cls_token=use_cls_token,
            use_max_pool=use_max_pool,
            use_mean_pool=use_mean_pool)

    def _init_parameters(self):
        with torch.no_grad():
            if self.use_cls_token:
                self.cls_token.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
            if self.use_pos_embs:
                self.pos_embs.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, imgs):
        '''
        Args:
            imgs: [batch*num_views, C, H, W]
        Return:
            output: [batch, num_classes]
        '''

        # img_feats: [batch*num_views, num_channels]
        #   NOTE here need to `squeeze` 
        img_feats = self.base_feature_extractor(imgs).squeeze()
        if 'resnet' in self.base_model_name and self.base_feature_dim != self.num_channels:
            img_feats = self.act(self.bn1d(self.linear_proj(img_feats)))

        # img_feats: [batch, num_views, num_channels]
        img_feats = img_feats.reshape(-1, self.num_views, self.num_channels)

        batch = img_feats.shape[0]
        if self.use_cls_token:  # NOTE use `cls_token` for classification
            # cls_token: [batch, 1, num_channels]
            cls_token = repeat(self.cls_token, 'd -> b n d', b=batch, n=1)
            # img_feats: [batch, 1+num_views, num_channels]
            img_feats = torch.cat([cls_token, img_feats], dim=1)
            if self.use_pos_embs:
                # pos_embs: [batch, 1+num_views, num_channels]
                pos_embs = repeat(self.pos_embs, '... -> b ...', b=batch)
                img_feats = img_feats + pos_embs
            
        else:   # NOTE do not use `cls_token` for classification
            if self.use_pos_embs:
                # pos_embs: [batch, num_views, num_channels]
                pos_embs = repeat(self.pos_embs, '... -> b ...', b=batch)
                img_feats = img_feats + pos_embs

        if self.vis_flag:
            # encoder_feats: [batch, num_views, num_channels] OR [batch, 1+num_views, num_channels]
            encoder_feats, attn_scores_list = self.encoder(img_feats)
        else:
            # encoder_feats: [batch, num_views, num_channels] OR [batch, 1+num_views, num_channels]
            encoder_feats = self.encoder(img_feats)
        # output: [batch, num_classes]
        output = self.cls_head(encoder_feats)

        if self.vis_flag:
            return output, attn_scores_list
        else:
            return output


class VSFormer_init_variant(nn.Module):
    '''
        Keep same as VSFormer other than `base_feature_extractor`, 
        this model uses a single Conv or two Convs to initialize input views
    '''
    def __init__(self,
        num_img_proj_layers: int = 1,
        num_layers: int = 6,
        num_heads: int = 6,
        num_channels: int = 384,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        max_dpr: float = 0.0,
        atten_drop: float = 0.0,
        mlp_drop: float = 0.0,
        num_views: int = 20,
        clshead_layers: int = 2,
        num_classes: int = 40,
        vis_flag: bool = False,
        activation_checkpointing: bool = False,):
        super().__init__()

        if num_img_proj_layers == 1:
            self.base_feature_extractor = nn.Sequential(
                # --- 1. refer to the first conv layer in ResNet18
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                # --- 2. adapt `dim_img_feat` to `num_channels`
                Rearrange('b c h w -> b (c h w)'),
                nn.Linear(56*56*64, num_channels)
            )
        elif num_img_proj_layers == 2:  # 没想到这种情况参数量远远少于 `num_img_proj_layers==1`
            self.base_feature_extractor = nn.Sequential(
                # --- 1. refer to the first conv layer in ResNet18
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                # --- 2. further convolute on the output of in step 1
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                # --- 3. adapt `dim_img_feat` to `num_channels`
                Rearrange('b c h w -> b (c h w)'),
                nn.Linear(28*28*32, num_channels)
            )

        self.num_channels = num_channels
        self.num_views = num_views
        self.vis_flag = vis_flag

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            widening_factor=widening_factor,
            max_dpr=max_dpr,
            atten_drop=atten_drop,
            mlp_drop=mlp_drop,
            activation_checkpointing=activation_checkpointing,
            vis_flag=vis_flag)

        self.cls_head = ClsHead(
            num_layers=clshead_layers,
            num_channels=num_channels,
            num_classes=num_classes)

    def forward(self, imgs):
        '''
        Args:
            imgs: [batch*num_views, C, H, W]
        Return:
            output: [batch, num_classes]
        '''

        # img_feats: [batch*num_views, num_channels]
        #   NOTE here need to `squeeze` 
        img_feats = self.base_feature_extractor(imgs).squeeze()
        # img_feats: [batch, num_views, num_channels]
        img_feats = img_feats.reshape(-1, self.num_views, self.num_channels)

        if self.vis_flag:
            # encoder_feats: [batch, num_views, num_channels] OR [batch, 1+num_views, num_channels]
            encoder_feats, attn_scores_list = self.encoder(img_feats)
        else:
            # encoder_feats: [batch, num_views, num_channels]
            encoder_feats = self.encoder(img_feats)
        # output: [batch, num_classes]
        output = self.cls_head(encoder_feats)

        if self.vis_flag:
            return output, attn_scores_list
        else:
            return output