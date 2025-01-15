import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the second half of the input tensor by swapping and negating its parts.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim), where dim is even.

    Returns:
        torch.Tensor: Rotated tensor of the same shape as input.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embedding (RoPE) to queries and keys.

    Args:
        q (torch.Tensor): Queries tensor of shape (..., dim).
        k (torch.Tensor): Keys tensor of shape (..., dim).
        cos (torch.Tensor): Cosine values for RoPE, of shape (..., dim // 2).
        sin (torch.Tensor): Sine values for RoPE, of shape (..., dim // 2).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Transformed queries and keys tensors.
    """
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def apply_rope_x(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embedding (RoPE) to a single tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim).
        cos (torch.Tensor): Cosine values for RoPE, of shape (..., dim // 2).
        sin (torch.Tensor): Sine values for RoPE, of shape (..., dim // 2).

    Returns:
        torch.Tensor: Transformed tensor.
    """
    return (x * cos) + (rotate_half(x) * sin)
