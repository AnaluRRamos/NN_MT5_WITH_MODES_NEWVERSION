import torch
import pytorch_lightning as pl
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sacrebleu
from src.mode_config import ModeConfig
from src.loss_functions import entity_aware_loss, ner_auxiliary_loss, placeholder_loss
import logging

logging.basicConfig(level=logging.INFO)

class T5FineTuner(pl.LightningModule):
    def __init__(self, tokenizer, train_dataloader, val_dataloader, test_dataloader, learning_rate, target_max_length=32, mode=0, num_ne_tags=17):
        super(T5FineTuner, self).__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.model = T5ForConditionalGeneration.from_pretrained(tokenizer.name_or_path)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.target_max_length = target_max_length
        self.mode = mode
        self.ne_tag_embedding = nn.Embedding(num_ne_tags, self.model.config.d_model)
        self.model.gradient_checkpointing_enable()

        # Log model parameters for verification
        logging.info(f"Initialized T5FineTuner with mode={mode}, learning_rate={learning_rate}, target_max_length={target_max_length}")
        logging.info(f"NE Tag Embedding Shape: {self.ne_tag_embedding.weight.shape}")

    def forward(self, source_token_ids, source_mask, target_token_ids=None, target_mask=None, ne_tag_mask=None, training=False):
        logging.debug(f"Forward pass called with training={training}")
        
        if training and self.mode == 3:
            outputs = self.model(input_ids=source_token_ids, attention_mask=source_mask, labels=target_token_ids, output_attentions=True, return_dict=True)
            lm_logits = outputs.logits
            cross_attentions = outputs.cross_attentions
            attention_weights = cross_attentions[-1].mean(dim=1)
            avg_attention = attention_weights.mean(dim=1)
            base_loss = outputs.loss
            loss = base_loss + ner_auxiliary_loss(avg_attention, ne_tag_mask) * ModeConfig.MODE_3_NER_WEIGHT
            return loss
        else:
            encoder_outputs = self.model.encoder(input_ids=source_token_ids, attention_mask=source_mask, return_dict=True)
            token_embeddings = encoder_outputs.last_hidden_state

            if ne_tag_mask is not None and self.mode != 0:
                ne_embeddings = self.ne_tag_embedding(ne_tag_mask)
                combined_embeddings = token_embeddings + ne_embeddings
                decoder_outputs = self.model.decoder(inputs_embeds=combined_embeddings, attention_mask=source_mask, return_dict=True)
            else:
                decoder_outputs = self.model.decoder(inputs_embeds=token_embeddings, attention_mask=source_mask, return_dict=True)

            lm_logits = self.model.lm_head(decoder_outputs.last_hidden_state)

            if training:
                if self.mode == 1:
                    loss = entity_aware_loss(lm_logits, target_token_ids, ne_tag_mask, weight_factor=ModeConfig.MODE_1_WEIGHT)
                elif self.mode == 2:
                    base_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(lm_logits.view(-1, lm_logits.size(-1)), target_token_ids.view(-1))
                    loss = placeholder_loss(base_loss, ne_tag_mask) * ModeConfig.MODE_2_WEIGHT
                else:
                    loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(lm_logits.view(-1, lm_logits.size(-1)), target_token_ids.view(-1)) * ModeConfig.MODE_0_WEIGHT
                return loss
            else:
                predicted_token_ids = self.model.generate(input_ids=source_token_ids, attention_mask=source_mask, max_length=self.target_max_length)
                return predicted_token_ids

    def translate(self, input_text):
        """Method to translate a given text."""
        self.eval()
        tokenized_input = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.target_max_length
        )
        input_ids = tokenized_input.input_ids.to(self.device)
        attention_mask = tokenized_input.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_length=self.target_max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        self.train()
        source_token_ids, source_mask, target_token_ids, target_mask, source_ne_tags = batch
        loss = self(source_token_ids, source_mask, target_token_ids, target_mask, ne_tag_mask=source_ne_tags, training=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        logging.debug(f"Training step {batch_idx}: Loss = {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        source_token_ids, source_mask, target_token_ids, target_mask, source_ne_tags = batch
        val_loss = self(source_token_ids, source_mask, target_token_ids, target_mask, ne_tag_mask=source_ne_tags, training=True)
        pred_token_ids = self(source_token_ids, source_mask)

        # Handle decoding and filtering valid ids
        def filter_valid_ids(token_ids):
            return [id for id in token_ids if 0 <= id < self.tokenizer.vocab_size]

        pred_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in pred_token_ids]
        target_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in target_token_ids]

        bleu_score = sacrebleu.corpus_bleu(pred_texts, [target_texts]).score

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_bleu', bleu_score, prog_bar=True)

        logging.info(f"Validation step {batch_idx}: BLEU Score = {bleu_score}")
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        logging.info(f"Optimizer configured: AdamW with learning rate = {self.learning_rate}")
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
