import numpy as np


def create_random_matrix(rows: int, cols: int) -> np.array:
    """Creates a random matrix of shape (rows, cols) with values drawn from a standard normal distribution."""
    random_matrix = np.random.randn(rows, cols) / np.sqrt(rows)
    return random_matrix


def generate_qkv(rows: int, cols: int) -> dict:
    """Generates random matrices for Q (Query), K (Key), and V (Value) with the specified number of rows and columns."""
    res = {}
    for c in ["Q", "K", "V"]:
        res[c] = create_random_matrix(rows, cols)
    return res


def embeding_matrix_qkv(x:np.array, W: dict) -> dict:
    """Computes the Q, K, and V matrices for the input x (token embeddings + possitions) using the wight matrices."""
    res = {}
    for c in ["Q", "K", "V"]:
        res[c] = x @ W[c]
    return res


def split_heads(matrix: np.array, heads: int, heads_dim: int) -> np.array:
    """Splits the input matrix into multiple heads for multi-head attention."""
    res = matrix.reshape(matrix.shape[0], heads, heads_dim)
    return res.transpose(1, 0, 2)


def combine_heads(matrix: np.array) -> np.array:
    """Combines the multiple heads back into a single matrix for multi-head attention."""
    return matrix.transpose(1, 0, 2).reshape(matrix.shape[1], -1)


def similarity_score(Q: np.array, K: np.array, d: int) -> np.array:
    """Computes the similarity score between the Q and K matrices using the scaled dot-product attention mechanism."""
    return (Q @ K.transpose(0, 2, 1)) / np.sqrt(d)


def softmax(matrix: np.array) -> np.array:
    """Applies the softmax function to the input matrix along the last axis."""
    max_vals = np.max(matrix, axis=-1, keepdims=True)
    exp_valores = np.exp(matrix - max_vals)
    suma_por_fila = np.sum(exp_valores, axis=-1, keepdims=True)
    softmax_result = exp_valores / suma_por_fila
    return softmax_result


def create_casual_mask(size: int) -> np.array:
    """Creates a causal mask for the attention mechanism, which prevents attending to future tokens."""
    mask = np.triu(np.full((size, size), -np.inf), k=1)
    return mask


def attention_forward(x:np.array, W: dict, heads: int) -> tuple:
    """Computes the attention output for the input x (token embeddings + positions) using the weight matrices."""
    heads_dim = x.shape[1] // heads
    qkt = embeding_matrix_qkv(x, W)

    # Split heads for multi-head attention
    qkt["Q"] = split_heads(qkt["Q"], heads, heads_dim)
    qkt["K"] = split_heads(qkt["K"], heads, heads_dim)
    qkt["V"] = split_heads(qkt["V"], heads, heads_dim)
    # Compute similarity scores and apply causal mask
    similarity_qk = similarity_score(qkt["Q"], qkt["K"], heads_dim)
    casual_mask = create_casual_mask(x.shape[0])
    similarity_qk += casual_mask
    # Apply softmax to obtain attention weights and compute the final attention output
    attention_weights = softmax(similarity_qk)
    attention = attention_weights @ qkt["V"]
    # Combine heads to get the final attention output
    attention = combine_heads(attention)
    return attention, attention_weights


def backward_attention(d_attention: np.array,x: np.array, attention_weights: np.array,
                        W: dict, heads: int, learning_rate: float) -> tuple:
    """Calculates gradients for the Multi-Head Attention layer, updates W (Q, K, V) and retunrs the gradient witch respect to x."""
    seq_len,embedding_dim= x.shape[0]
    heads_dim = embedding_dim // heads
    fragment_embedding = embedding_matrix[fragment]
    qkv = embeding_matrix_qkv(x, W)

    Q = split_heads(qkv["Q"], heads, heads_dim)
    K = split_heads(qkv["K"], heads, heads_dim)
    V = split_heads(qkv["V"], heads, heads_dim)

    d_combined = d_attention.reshape(seq_len, heads, heads_dim).transpose(1, 0, 2)

    d_V = attention_weights.transpose(0, 2, 1) @ d_combined
    d_attention_weights = d_combined @ V.transpose(0, 2, 1)

    sum_d_a = np.sum(d_attention_weights * attention_weights, axis=-1, keepdims=True)
    d_similarity = attention_weights * (d_attention_weights - sum_d_a)

    d_similarity = d_similarity / np.sqrt(heads_dim)
    d_Q = d_similarity @ K
    d_K = d_similarity.transpose(0, 2, 1) @ Q

    dQ_full = d_Q.transpose(1, 0, 2).reshape(seq_len, embedding_dim)
    dK_full = d_K.transpose(1, 0, 2).reshape(seq_len, embedding_dim)
    dV_full = d_V.transpose(1, 0, 2).reshape(seq_len, embedding_dim)

    dW = {
        "Q": x.T @ dQ_full,
        "K": x.T @ dK_full,
        "V": x.T @ dV_full
    }

    d_x = (dQ_full @ W["Q"].T) + (dK_full @ W["K"].T) + (dV_full @ W["V"].T)

    for c in ["Q", "K", "V"]:
        W[c] -= learning_rate * dW[c]

    return W, d_x
