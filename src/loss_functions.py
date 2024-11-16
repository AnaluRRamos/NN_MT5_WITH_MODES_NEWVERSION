import torch
from torch.nn import CrossEntropyLoss

def entity_aware_loss(logits, labels, ne_tag_mask, weight_factor=2.0):
    """
    Calculate the entity-aware loss by adding additional weights to named entity tokens.
    
    Args:
        logits (torch.Tensor): The model output logits (batch_size, seq_length, vocab_size).
        labels (torch.Tensor): The ground truth labels (batch_size, seq_length).
        ne_tag_mask (torch.Tensor): Mask indicating positions of named entities (batch_size, seq_length).
        weight_factor (float): The weight to apply for named entity tokens. Default is 2.0.

    Returns:
        torch.Tensor: The calculated entity-aware loss.
    """
    # Shifting logits and labels for teacher forcing (ignore the last token prediction)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Basic cross-entropy loss (element-wise loss without reduction)
    loss_fct = CrossEntropyLoss(reduction="none")
    base_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Apply extra weight to NE tags in the loss
    entity_weights = torch.ones_like(shift_labels).float()
    entity_weights[ne_tag_mask == 1] *= weight_factor
    weighted_loss = base_loss * entity_weights.view(-1)

    return weighted_loss.mean()

def ner_auxiliary_loss(attention_weights, ne_tag_mask, ner_weight=1.0):
    """
    Calculate the auxiliary NER loss by comparing attention weights with NE tags.
    
    Args:
        attention_weights (torch.Tensor): Attention weights from the model (batch_size, seq_length).
        ne_tag_mask (torch.Tensor): Mask indicating positions of named entities (batch_size, seq_length).
        ner_weight (float): Weight applied to the NER loss. Default is 1.0.

    Returns:
        torch.Tensor: The calculated NER auxiliary loss.
    """
    avg_attention = attention_weights.mean(dim=-1)
    ner_loss = torch.mean((avg_attention - ne_tag_mask.float()) ** 2) * ner_weight
    return ner_loss

def placeholder_loss(base_loss, ne_tag_mask, placeholder_weight=1.0):
    """
    Calculate the placeholder loss by applying the NE tag mask's average value to the base loss.
    
    Args:
        base_loss (torch.Tensor): The base loss calculated using cross-entropy.
        ne_tag_mask (torch.Tensor): Mask indicating positions of named entities (batch_size, seq_length).
        placeholder_weight (float): Weight factor for adjusting the placeholder loss effect. Default is 1.0.

    Returns:
        torch.Tensor: The calculated placeholder loss.
    """
    return base_loss * torch.mean(ne_tag_mask.float()) * placeholder_weight
