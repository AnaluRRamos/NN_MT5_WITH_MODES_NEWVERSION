import torch
from torch import nn

def gradual_weight_leaky(scaling, negative_slope=0.1, target_weight=1.5):
    """
    Computes a gradual weight factor using Leaky ReLU.
    
    The input 'scaling' is expected to be a value between 0 and 1 (e.g. current_epoch / warmup_epochs).
    This function first centers scaling around zero (so that scaling=0 becomes -0.5 and scaling=1 becomes 0.5),
    applies a Leaky ReLU activation, shifts the output back, normalizes it to [0,1], and then maps it to [1, target_weight].
    
    Parameters:
      scaling (float): A value between 0 and 1.
      negative_slope (float): The negative slope for LeakyReLU.
      target_weight (float): The final weight factor when scaling is 1.
      
    Returns:
      float: A weight factor between 1 and target_weight.
    """
    # Center scaling so that it goes from -0.5 to 0.5.
    x = scaling - 0.5
    # Apply Leaky ReLU.
    activated = nn.LeakyReLU(negative_slope=negative_slope)(torch.tensor(x))
    # Shift back by adding 0.5. Now 'adjusted' is roughly in [0.5, 1.0].
    adjusted = activated + 0.5
    # Normalize: Map [0.5, 1.0] to [0, 1]
    normalized = (adjusted - 0.5) * 2
    # Map normalized value to [1, target_weight]
    weight_factor = 1 + normalized.item() * (target_weight - 1)
    return weight_factor


def entity_aware_loss(logits, labels, ne_tag_mask, weight_factor=2.0):
    """
    Computes the entity-aware loss by applying a higher weight to named entity tokens.
    
    Args:
        logits (Tensor): Model output logits (batch_size, seq_len, vocab_size).
        labels (Tensor): Target token IDs (batch_size, seq_len).
        ne_tag_mask (Tensor): Binary mask (1 for NE tokens, 0 otherwise) (batch_size, seq_len).
        weight_factor (float): Weighting factor for NE-tagged tokens.

    Returns:
        Tensor: Weighted cross-entropy loss.
    """
    logits_flat = logits.view(-1, logits.size(-1))  # Flatten logits
    labels_flat = labels.view(-1)  # Flatten labels
    ne_tag_mask_flat = ne_tag_mask.view(-1)  # Flatten NE mask

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fn(logits_flat, labels_flat)

    weights = torch.ones_like(loss)  # Default weight of 1
    weights[ne_tag_mask_flat != 0] *= weight_factor  # Apply higher weight to NE tokens

    weighted_loss = loss * weights  # Apply the weighting
    return weighted_loss.mean()


def ner_auxiliary_loss(attention_weights, ne_tag_mask):
    """
    Auxiliary loss to ensure the model attends more to named entities.

    Args:
        attention_weights (Tensor): Attention scores (batch_size, seq_len).
        ne_tag_mask (Tensor): Binary mask (1 for NE tokens, 0 otherwise) (batch_size, seq_len).

    Returns:
        Tensor: Mean squared error loss between attention and NE tags.
    """
    avg_attention = attention_weights.mean(dim=-1)  # Average attention over heads
    ner_loss = torch.mean((avg_attention - ne_tag_mask.float()) ** 2)
    return ner_loss


def placeholder_loss(base_loss, ne_tag_mask):
    """
    Adjusts the base loss by scaling it according to the proportion of NE tags.

    Args:
        base_loss (Tensor): The base loss computed without NE weighting.
        ne_tag_mask (Tensor): Binary mask (1 for NE tokens, 0 otherwise) (batch_size, seq_len).

    Returns:
        Tensor: Scaled loss.
    """
    ne_tag_mask_flat = ne_tag_mask.view(-1).float()
    if torch.mean(ne_tag_mask_flat) == 0:  # Avoid division by zero
        return base_loss
    scaled_loss = base_loss * torch.mean(ne_tag_mask_flat)
    return scaled_loss
