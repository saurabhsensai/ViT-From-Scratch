import torch 
import torch.nn as nn
from einops import rearrange, repeat 
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim= 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h  w) (p1 p2 c)', p1 = patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)

        )

    def forward(self, x):
        x = self.projection(x)
        return x

'''
Checking the Dimentions

data = torch.randn(4, 3, 224, 224)
print(data.shape)

patch = PatchEmbedding()
out = patch(data)

print(out.shape)

'''

class Attention(nn.Module):
    def __init__(self, dim,  num_heads=8, qkv_bias= False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 
        self.qkv = nn.Linear(dim, dim*3, bias= qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] , qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=1)
        self.attention_weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


'''

data = torch.rand(1, 196, 768)
attn = Attention(768)

out = attn(data)
print(out.shape)
'''

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x 


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qvk_bias=False, drop=0, attn_drop=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qvk_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x 

'''
data = torch.rand(1, 196, 768)
blc = Block(768, 8)

out = blc(data)
print(out.shape)

'''


class VisionTransformer(nn.Module):
    def __init__(self, image_size= 224, patch_size=16, in_channels=3, num_classes=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                    qvk_bias=True, drop_rate= 0.1, attn_drop_rate=0):
                    super().__init__()
                    self.image_size = image_size
                    self.patch_size = patch_size
                    self.num_pathces = (image_size // patch_size) ** 2
                    self.num_classes = num_classes
                    self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                    self.pos_embed = nn.Parameter(torch.zeros(1, self.num_pathces + 1, embed_dim))
                    self.pos_drop = nn.Dropout(p=drop_rate)
                    self.blocks = nn.ModuleList([
                        Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qvk_bias=qvk_bias,
                        drop=drop_rate, attn_drop=attn_drop_rate)
                        for _ in range(depth)
                    ])
                    self.norm = nn.LayerNorm(embed_dim)
                    self.head = nn.Linear(embed_dim, num_classes)
                    self.initialize_weights()


    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        self.original_images = x 
        B = x.shape[0]
        x = self.patch_embed(x)
        self.patch_embedding = x
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == len(self.blocks) - 1:
                self.last_attn_weights = block.attn.attention_weights
        self.token_embeddings = x
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x 










