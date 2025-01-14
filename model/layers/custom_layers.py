import torch


class CustomLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initializes a custom linear layer.

        Args:
            input_size (int): The size of the input features.
            output_size (int): The size of the output features.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(0.01 * torch.randn((output_size, input_size)))
        self.bias = torch.nn.Parameter(torch.zeros((output_size,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the linear transformation: y = x @ weight.T + bias.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return x @ self.weight.T + self.bias


class CustomEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        """
        Initializes a custom embedding layer.

        Args:
            num_embeddings (int): The number of embeddings (vocabulary size).
            embedding_dim (int): The dimension of each embedding vector.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            0.01 * torch.randn((num_embeddings, embedding_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps input indices to their corresponding embedding vectors.

        Args:
            x (torch.Tensor): Input tensor containing indices, of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor containing embedding vectors, of shape (batch_size, seq_len, embedding_dim).
        """
        return self.weight[x]
