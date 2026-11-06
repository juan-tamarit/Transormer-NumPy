import numpy as np


def relu(matrix: np.array) -> np.array:
    """Applies the ReLU (n>0?n:0) activation function to the input matrix."""
    return np.maximum(matrix, 0)


def feed_forward(attention_matrix: np.array, expansion_matrix: np.array, compresion_matrix: np.array) -> np.array:
    """Applies a feed-forward neural network to the attention output using ReLU activation."""
    # We are using relu as the activation function because it is common and cheap to implement, Gelu usually gives better results but is more expensive to compute.
    attention_relu = relu(attention_matrix @ expansion_matrix) @ compresion_matrix
    return attention_relu


def backward_feed_forward(d_final_output: np.array, attention_matrix: np.array, W_ffn: dict,
                           learning_rate: float) -> tuple:
    """Performs the backward pass for the feed-forward neural network, updating the expansion and compression matrices and returning the gradient with respect to the attention matrix."""
    expansion_matrix = W_ffn["expansion_matrix"]
    compression_matrix = W_ffn["compression_matrix"]

    hidden_linear = attention_matrix @ expansion_matrix
    hidden_linear_relu = relu(hidden_linear)

    d_compression_matrix = hidden_linear_relu.T @ d_final_output

    d_hidden_linear_relu = d_final_output @ compression_matrix.T
    d_hidden_linear = d_hidden_linear_relu * (hidden_linear > 0)

    d_expansion_matrix = attention_matrix.T @ d_hidden_linear

    d_attention_matrix = d_hidden_linear @ expansion_matrix.T

    W_ffn["expansion_matrix"] -= learning_rate * d_expansion_matrix
    W_ffn["compression_matrix"] -= learning_rate * d_compression_matrix

    return d_attention_matrix, W_ffn
