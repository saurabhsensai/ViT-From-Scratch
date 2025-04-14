# vit/visualizer.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class ViTVisualizer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def visualize_patches(self, batch_idx=0, sample_idx=0, save_path=None):
        images, _ = next(iter(self.dataloader))
        image = images[sample_idx].permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        patch_size = self.model.patch_size
        h, w = image.shape[0], image.shape[1]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        for i in range(0, h, patch_size):
            ax.axhline(i, color='white', lw=0.5)
        for j in range(0, w, patch_size):
            ax.axvline(j, color='white', lw=0.5)
        plt.title(f"Image divided into {(h//patch_size)*(w//patch_size)} patches of size {patch_size}x{patch_size}")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def visualize_embeddings(self, layer='patch', batch_idx=0, save_path=None):
        images, labels = next(iter(self.dataloader))
        images = images.to(self.device)
        self.model.eval()
        with torch.no_grad():
            _ = self.model(images)
        if layer == 'patch':
            embeddings = self.model.patch_embeddings.cpu().numpy()
            title = "Patch Embeddings"
        elif layer == 'token':
            embeddings = self.model.token_embeddings[:, 1:].cpu().numpy()
            title = "Token Embeddings (after transformer)"
        else:
            raise ValueError("Layer must be either 'patch' or 'token'")
        embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], -1)
        embedding_labels = np.repeat(labels.cpu().numpy(), embeddings.shape[0] // len(labels))
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=embedding_labels,
                             cmap='tab10', alpha=0.8, s=10)
        plt.colorbar(scatter, label='Class')
        plt.title(f"t-SNE visualization of {title}")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def visualize_attention(self, sample_idx=0, head_idx=0, save_path=None):
        images, _ = next(iter(self.dataloader))
        image = images[sample_idx:sample_idx+1].to(self.device)
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        attn_weights = self.model.last_attn_weights[0, head_idx].cpu()
        attn_from_cls = attn_weights[0, 1:].reshape(int(np.sqrt(self.model.num_patches)),
                                                   int(np.sqrt(self.model.num_patches)))
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        im = axs[1].imshow(attn_from_cls, cmap='hot')
        axs[1].set_title(f"Attention Map (Head {head_idx})")
        axs[1].axis('off')
        fig.colorbar(im, ax=axs[1], shrink=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def overlay_attention(self, sample_idx=0, head_idx=0, save_path=None):
        images, _ = next(iter(self.dataloader))
        image = images[sample_idx:sample_idx+1].to(self.device)
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        attn_weights = self.model.last_attn_weights[0, head_idx].cpu()
        attn_from_cls = attn_weights[0, 1:].reshape(int(np.sqrt(self.model.num_patches)),
                                                   int(np.sqrt(self.model.num_patches)))
        attn_resized = F.interpolate(attn_from_cls.unsqueeze(0).unsqueeze(0),
                                     size=(image.shape[2], image.shape[3]),
                                     mode='bilinear', align_corners=False)
        attn_resized = attn_resized[0, 0].cpu().numpy()
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        axs[0].imshow(img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        im = axs[1].imshow(attn_resized, cmap='hot')
        axs[1].set_title(f"Attention Map (Head {head_idx})")
        axs[1].axis('off')
        fig.colorbar(im, ax=axs[1], shrink=0.6)
        axs[2].imshow(img)
        axs[2].imshow(attn_resized, alpha=0.5, cmap='hot')
        axs[2].set_title("Attention Overlay")
        axs[2].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def visualize_all_heads(self, sample_idx=0, save_path=None):
        images, _ = next(iter(self.dataloader))
        image = images[sample_idx:sample_idx+1].to(self.device)
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        attn_weights = self.model.last_attn_weights[0].cpu()
        num_heads = attn_weights.shape[0]
        fig, axs = plt.subplots(int(np.ceil(num_heads / 4)), 4, figsize=(20, 5 * int(np.ceil(num_heads / 4))))
        if num_heads <= 4:
            axs = axs.reshape(1, -1)
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        for head_idx in range(num_heads):
            row, col = head_idx // 4, head_idx % 4
            attn_from_cls = attn_weights[head_idx, 0, 1:].reshape(int(np.sqrt(self.model.num_patches)),
                                                               int(np.sqrt(self.model.num_patches)))
            attn_resized = F.interpolate(attn_from_cls.unsqueeze(0).unsqueeze(0),
                                         size=(image.shape[2], image.shape[3]),
                                         mode='bilinear', align_corners=False)
            attn_resized = attn_resized[0, 0].cpu().numpy()
            axs[row, col].imshow(img)
            axs[row, col].imshow(attn_resized, alpha=0.5, cmap='hot')
            axs[row, col].set_title(f"Head {head_idx}")
            axs[row, col].axis('off')
        for i in range(num_heads, axs.shape[0] * axs.shape[1]):
            row, col = i // 4, i % 4
            axs[row, col].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()