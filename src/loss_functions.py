import torch
import torch.nn as nn

def entity_aware_loss(logits, labels, ne_tag_mask, weight_factor=2.0):
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    ne_tag_mask_flat = ne_tag_mask.view(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fn(logits_flat, labels_flat)
    weights = torch.ones_like(loss)
    weights[ne_tag_mask_flat != 0] *= weight_factor
    weighted_loss = loss * weights
    return weighted_loss.mean()

def ner_auxiliary_loss(attention_weights, ne_tag_mask):
    avg_attention = attention_weights.mean(dim=-1)
    ner_loss = torch.mean((avg_attention - ne_tag_mask.float()) ** 2)
    return ner_loss

def placeholder_loss(base_loss, ne_tag_mask):
    ne_tag_mask_flat = ne_tag_mask.view(-1).float()
    scaled_loss = base_loss * torch.mean(ne_tag_mask_flat)
    return scaled_loss
