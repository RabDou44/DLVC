


## TODO implement your own ViT in this file
# You can take any existing code from any repository or blog post - it doesn't have to be a huge model
# specify from where you got the code and integrate it into this code repository so that 
# you can run the model with this code
import torch.nn as nn
import torch.nn.functional as F
import torch


class ViTClassifierModel(nn.Module):
    """ViT Model for Image Classification."""

    def __init__(self, num_transformer_layers=3, mlp_head_units=None, num_classes=10, patch_size=None, num_patches=None,
                 batch_size = None, embed_dim=None, device = None, num_heads=None, key_dim=None, ff_dim=None):
        """Init Function."""
        super().__init__()
        if mlp_head_units is None:
            mlp_head_units = [512, 256]
        self.create_patch_layer = CreatePatchesLayer(patch_size, patch_size)
        self.patch_embedding_layer = PatchEmbeddingLayer(num_patches, batch_size, patch_size, embed_dim, device)
        self.transformer_layers = torch.nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerBlock(
                    num_heads, key_dim, embed_dim, ff_dim
                )
            )

        self.mlp_block = create_mlp_block(
            input_features=192,
            output_features=[512, 256],
            activation_function=torch.nn.GELU,
            dropout_rate=0.5,
        )

        self.logits_layer = torch.nn.Linear(mlp_head_units[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        x = self.create_patch_layer(x)
        x = self.patch_embedding_layer(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = x[:, 0]
        x = self.mlp_block(x)
        x = self.logits_layer(x)
        return x


class CreatePatchesLayer(torch.nn.Module):
    """Custom PyTorch Layer to Extract Patches from Images."""

    def __init__(
            self,
            patch_size: int,
            strides: int,
    ) -> None:
        """Init Variables."""
        super().__init__()
        self.unfold_layer = torch.nn.Unfold(
            kernel_size=patch_size, stride=strides
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward Pass to Create Patches."""
        patched_images = self.unfold_layer(images)
        return patched_images.permute((0, 2, 1))


class PatchEmbeddingLayer(torch.nn.Module):
    """Positional Embedding Layer for Images of Patches."""

    def __init__(
            self,
            num_patches: int,
            batch_size: int,
            patch_size: int,
            embed_dim: int,
            device: torch.device,
    ) -> None:
        """Init Function."""
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.position_emb = torch.nn.Embedding(
            num_embeddings=num_patches + 1, embedding_dim=embed_dim
        )
        self.projection_layer = torch.nn.Linear(
            patch_size * patch_size * 3, embed_dim
        )
        self.class_parameter = torch.nn.Parameter(
            torch.randn(1, 1, embed_dim).to(device),
            requires_grad=True,
        )
        self.device = device

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        batch_size = patches.size(0)

        # Expand class token to match the actual batch size
        class_token = self.class_parameter.expand(batch_size, -1, -1)  # shape: (B, 1, D)

        patches = self.projection_layer(patches)  # shape: (B, num_patches, D)

        # Concatenate class token
        x = torch.cat((class_token, patches), dim=1)  # shape: (B, num_patches+1, D)

        # Add positional embedding
        positions = torch.arange(0, self.num_patches + 1).to(self.device).unsqueeze(0)  # shape: (1, num_patches+1)
        x = x + self.position_emb(positions)  # broadcasting over batch

        return x

def create_mlp_block(input_features, output_features, activation_function, dropout_rate):
  layers = []
  in_features = input_features
  for i, out_features in enumerate(output_features):
    layers.append(nn.Linear(in_features, out_features))
    if i < len(output_features) - 1:  # No activation/dropout after last layer
      layers.append(activation_function())
      if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    in_features = out_features
  return nn.Sequential(*layers)


class TransformerBlock(torch.nn.Module):
    """Transformer Block Layer."""

    def __init__(
            self,
            num_heads: int,
            key_dim: int,
            embed_dim: int,
            ff_dim: int,
            dropout_rate: float = 0.1,
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.layer_norm_input = torch.nn.LayerNorm(
            normalized_shape=embed_dim, eps=1e-6
        )
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim,
            batch_first=True,
        )

        self.dropout_1 = torch.nn.Dropout(p=dropout_rate)
        self.layer_norm_1 = torch.nn.LayerNorm(
            normalized_shape=embed_dim, eps=1e-6
        )
        self.layer_norm_2 = torch.nn.LayerNorm(
            normalized_shape=embed_dim, eps=1e-6
        )
        self.ffn = create_mlp_block(
            input_features=embed_dim,
            output_features=[ff_dim, embed_dim],
            activation_function=torch.nn.GELU,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        layer_norm_inputs = self.layer_norm_input(inputs)
        attention_output, _ = self.attn(
            query=layer_norm_inputs,
            key=layer_norm_inputs,
            value=layer_norm_inputs,
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.layer_norm_1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        output = self.layer_norm_2(out1 + ffn_output)
        return output

