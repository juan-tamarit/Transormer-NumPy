import numpy as np

from attention import softmax


def lm_head(final_output: np.array, W_out: np.array) -> np.array:
    """Applies a linear transformation to the final output to produce logits for language modeling."""
    logits = final_output @ W_out
    return logits


def cross_entropy_loss(probabilities: np.array, targets: list) -> float:
    """Calculates the cross-entropy loss between the predicted probabilities and the true targets."""
    correct_probs = probabilities[np.arange(len(targets)), targets]
    clip_probs = np.clip(correct_probs, 1e-10, 1.0)
    loss = -np.mean(np.log(clip_probs))
    return float(loss)


def backward_loss_and_softmax(logits: np.array, targets: list) -> np.array:
    """Calculates the gradient of the loss with respect to the logits."""
    probabilities = softmax(logits)
    targets_one_hot = np.zeros_like(probabilities)
    targets_one_hot[np.arange(len(targets)), targets] = 1.0
    dlogits = (probabilities - targets_one_hot)
    return dlogits


def backward_lm_head(dlogits: np.array, final_output: np.array, W_out: np.array, learning_rate: float) -> tuple:
    """Performs the backward pass for the language modeling head, updating the output weights and returning the gradient with respect to the final output."""
    dW_out = final_output.T @ dlogits
    d_final_output = dlogits @ W_out.T
    W_out -= learning_rate * dW_out
    return d_final_output, W_out
