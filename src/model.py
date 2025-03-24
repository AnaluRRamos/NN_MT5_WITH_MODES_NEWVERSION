import torch
import pytorch_lightning as pl
from torch import nn
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
import sacrebleu
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk  # Ensure correct import
from src.mode_config import ModeConfig
from src.loss_functions import entity_aware_loss, ner_auxiliary_loss, placeholder_loss
import logging
import os
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.loggers import CometLogger # changed 13/03
#from pytorch_lightning.loggers import WandbLogger

from torchmetrics.text.bleu import BLEUScore

# Download NLTK METEOR data (only needed once)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

comet_logger = CometLogger(
    api_key="HvL5dz50hej2GlMfczDmW99Hk",  
    project_name="your_project_name"
)

logging.basicConfig(level=logging.INFO)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, tokenizer, train_dataloader, val_dataloader, test_dataloader, learning_rate, target_max_length=600, mode=0, num_ne_tags=26):
        super(MT5FineTuner, self).__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.target_max_length = target_max_length
        self.mode = mode
        self.dropout = nn.Dropout(0.1)  # Dropout addition   
        self.ne_tag_embedding = nn.Embedding(num_ne_tags, self.model.config.d_model)
        self.model.gradient_checkpointing_enable()
        self.all_test_preds = []
        
        logging.info(f"Initialized MT5FineTuner with mode={mode}, learning_rate={learning_rate}, target_max_length={target_max_length}")
        logging.info(f"NE Tag Embedding Shape: {self.ne_tag_embedding.weight.shape}")

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Initialize BLEU scorer using sacrebleu
        #self.metric_bleu = sacrebleu.metrics.BLEU() changes 11/03
        self.metric_bleu = None  # sacrebleu is used at the sentence level now


    
    def forward(self, source_token_ids, source_mask, target_token_ids=None, target_mask=None, ne_tag_mask=None, training=False):
        if training:
            labels = target_token_ids.clone()
            labels[target_token_ids == self.tokenizer.pad_token_id] = -100
            outputs = self.model(
                input_ids=source_token_ids,
                attention_mask=source_mask,
                labels=labels,
                return_dict=True
            )
            lm_logits = outputs.logits
            #lm_logits = self.dropout(lm_logits)  # This applies standard random dropout to the logits

            if self.mode == 0:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100,label_smoothing=0.1)
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            

            ###CODE CHANGED: Gradual weight with leaky rely 
            elif self.mode == 1:
                warmup_epochs = getattr(ModeConfig, "WARMUP_EPOCHS", 5)
                scaling = min(1.0, self.current_epoch / warmup_epochs)
                current_weight_factor = gradual_weight_leaky(scaling, negative_slope=0.1, target_weight=ModeConfig.MODE_1_WEIGHT)
    
                label_smoothing_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                lm_logits_flat = lm_logits.view(-1, lm_logits.size(-1))
                labels_flat = labels.view(-1)
                non_ner_mask = (ne_tag_mask.view(-1) == 0)
                non_ner_loss = label_smoothing_loss_fn(lm_logits_flat[non_ner_mask], labels_flat[non_ner_mask])
    
                ner_loss = entity_aware_loss(lm_logits, labels, ne_tag_mask, weight_factor=current_weight_factor)
    
                loss = non_ner_loss + ner_loss


            elif self.mode == 2:
                base_loss = nn.CrossEntropyLoss(ignore_index=-100)(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = placeholder_loss(base_loss, ne_tag_mask) * ModeConfig.MODE_2_WEIGHT
            else:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)) * ModeConfig.MODE_0_WEIGHT
            return loss
        else:
            # CHANGED: Use self.target_max_length from configuration for consistency.
            generated_token_ids = self.model.generate(
                input_ids=source_token_ids,
                attention_mask=source_mask,
                max_length=self.target_max_length,
                num_beams=8,
                early_stopping=True
            )
            return generated_token_ids

    def training_step(self, batch, batch_idx):
        source_token_ids, source_mask, source_ne_tags, target_token_ids, target_mask = batch
        loss = self(
            source_token_ids.to(self.device),
            source_mask.to(self.device),
            target_token_ids=target_token_ids.to(self.device),
            target_mask=target_mask.to(self.device),
            ne_tag_mask=source_ne_tags.to(self.device),
            training=True
        )
        # CHANGED: Added sync_dist=True for distributed logging.
       # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_token_ids, source_mask, source_ne_tags, target_token_ids, target_mask = batch
        # CHANGED: Ensure consistent device assignment using self.device
        val_loss = self(
            source_token_ids.to(self.device),
            source_mask.to(self.device),
            target_token_ids=target_token_ids.to(self.device),
            target_mask=target_mask.to(self.device),
            ne_tag_mask=source_ne_tags.to(self.device),
            training=True
        )
        pred_token_ids = self(
            source_token_ids.to(self.device),
            source_mask.to(self.device),
            training=False
        )
        
        def filter_valid_ids(token_ids):
            return [id for id in token_ids if 0 <= id < self.tokenizer.vocab_size]
        
        pred_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in pred_token_ids]
        target_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in target_token_ids]
        
        clean_pred_texts = [text.replace('<extra_id_0>', '').strip() for text in pred_texts]
        clean_target_texts = [text.replace('<extra_id_0>', '').strip() for text in target_texts]
        
        tokenized_pred_texts = [word_tokenize(pred) for pred in clean_pred_texts]
        tokenized_target_texts = [word_tokenize(ref) for ref in clean_target_texts]
        
        bleu_score = sacrebleu.corpus_bleu(clean_pred_texts, [clean_target_texts]).score
        
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        for pred, ref in zip(clean_pred_texts, clean_target_texts):
            scores = self.rouge_scorer.score(pred, ref)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        num_samples = len(clean_pred_texts)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples
        
        meteor_scores = [nltk.translate.meteor_score.meteor_score([ref], pred) for pred, ref in zip(tokenized_pred_texts, tokenized_target_texts)]
        meteor_score_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        
        #self.log('val_bleu', bleu_score, prog_bar=True, on_epoch=True, sync_dist=True)
        #self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_bleu', bleu_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rouge1', rouge_scores['rouge1'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rouge2', rouge_scores['rouge2'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rougeL', rouge_scores['rougeL'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_meteor', meteor_score_avg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        return val_loss
    
    def test_step(self, batch, batch_idx):
        source_token_ids, source_mask, ne_tags, target_ids, target_mask = batch

        pred_token_ids = self(
            source_token_ids.to(self.device),
            source_mask.to(self.device),
            training=False
        )

        def filter_valid_ids(token_ids):
            return [id for id in token_ids if 0 <= id < self.tokenizer.vocab_size]

        pred_texts = [
            self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True)
            for ids in pred_token_ids
        ]
        target_texts = [
            self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True)
            for ids in target_ids
        ]


        clean_pred_texts = [text.replace('<extra_id_0>', '').strip() for text in pred_texts]
        clean_target_texts = [text.replace('<extra_id_0>', '').strip() for text in target_texts]

    
        bleu_scores = [
            sacrebleu.sentence_bleu(pred, [ref]).score
            for pred, ref in zip(clean_pred_texts, clean_target_texts)
        ]

    
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        for pred, ref in zip(clean_pred_texts, clean_target_texts):
            scores = self.rouge_scorer.score(pred, ref)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        num_samples = len(clean_pred_texts)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples

  
        from nltk.tokenize import word_tokenize
        tokenized_pred_texts = [word_tokenize(pred) for pred in clean_pred_texts]
        tokenized_target_texts = [word_tokenize(ref) for ref in clean_target_texts]

        meteor_scores = [
            nltk.translate.meteor_score.meteor_score([ref], pred)
            for pred, ref in zip(tokenized_pred_texts, tokenized_target_texts)
        ]
        meteor_score_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

 
        for pred, ref, bleu in zip(clean_pred_texts, clean_target_texts, bleu_scores):
            self.all_test_preds.append({"prediction": pred, "reference": ref, "bleu": bleu})

   
        self.log('test_bleu', sum(bleu_scores) / len(bleu_scores), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_rouge1', rouge_scores['rouge1'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_rouge2', rouge_scores['rouge2'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_rougeL', rouge_scores['rougeL'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_meteor', meteor_score_avg, prog_bar=True, on_epoch=True, sync_dist=True)

        return {"pred_texts": clean_pred_texts, "bleu": bleu_scores, "rouge": rouge_scores, "meteor": meteor_score_avg}


    def on_test_epoch_end(self):
    # Sort by BLEU score (highest to lowest)
        sorted_preds = sorted(self.all_test_preds, key=lambda x: x["bleu"], reverse=True)
    
    # Get top 100 translations
        top_100_preds = sorted_preds[:100]

        file_name = f"top_100_test_translations_mode_{self.mode}.txt"

        with open(file_name, "w", encoding="utf-8") as f:
            for idx, entry in enumerate(top_100_preds, start=1):
                f.write(f"Rank: {idx}\n")
                f.write(f"BLEU Score: {entry['bleu']:.4f}\n")
                f.write(f"Prediction: {entry['prediction']}\n")
                f.write(f"Reference: {entry['reference']}\n\n")

    # Log average BLEU score across all test data
        avg_bleu = sum([entry["bleu"] for entry in self.all_test_preds]) / len(self.all_test_preds) if self.all_test_preds else 0.0
        self.log('test_bleu_avg', avg_bleu, prog_bar=True, on_epoch=True, sync_dist=True)

    
    

    # Tirei dia 11/02 para adicionar o teste selecionando traduçoes com melhores BLEUS

    """def test_step(self, batch, batch_idx):
        source_token_ids, source_mask, ne_tags, target_ids, target_mask = batch

        pred_token_ids = self(
            source_token_ids.to(self.device),
            source_mask.to(self.device),
            training=False
        )

        def filter_valid_ids(token_ids):
            return [id for id in token_ids if 0 <= id < self.tokenizer.vocab_size]

        pred_texts = [
            self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True)
            for ids in pred_token_ids
        ]
        target_texts = [
            self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True)
            for ids in target_ids
        ]

    # Clean texts
        clean_pred_texts = [text.replace('<extra_id_0>', '').strip() for text in pred_texts]
        clean_target_texts = [text.replace('<extra_id_0>', '').strip() for text in target_texts]

    # Compute BLEU score (using sacrebleu)
        bleu_score = sacrebleu.corpus_bleu(clean_pred_texts, [clean_target_texts]).score

    # Compute ROUGE using the rouge_score package
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        for pred, ref in zip(clean_pred_texts, clean_target_texts):
            scores = self.rouge_scorer.score(pred, ref)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        num_samples = len(clean_pred_texts)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples

    # For Meteor, ensure the texts are tokenized into lists of tokens.
        from nltk.tokenize import word_tokenize
        tokenized_pred_texts = [word_tokenize(pred) for pred in clean_pred_texts]
        tokenized_target_texts = [word_tokenize(ref) for ref in clean_target_texts]

        meteor_scores = [
            nltk.translate.meteor_score.meteor_score([ref], pred)
            for pred, ref in zip(tokenized_pred_texts, tokenized_target_texts)
        ]
        meteor_score_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

        self.log('test_bleu', bleu_score, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_rouge1', rouge_scores['rouge1'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_rouge2', rouge_scores['rouge2'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_rougeL', rouge_scores['rougeL'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_meteor', meteor_score_avg, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        self.all_test_preds.extend(clean_pred_texts)
    # Return the predictions along with metrics so they can be aggregated later.
        return {"pred_texts": clean_pred_texts, "bleu": bleu_score, "rouge": rouge_scores, "meteor": meteor_score_avg}


    def on_test_epoch_end(self):
       
        all_entries = []
        for idx, translation in enumerate(self.all_test_preds):
        
            article_id = f"article_{idx}"
            entry = f"Article ID: {article_id} (Mode: {self.mode})\nTranslation: {translation}"
            all_entries.append(entry)

    
        file_name = f"test_translations_mode_{self.mode}.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(entry + "\n")
    
    
        self.log("saved_test_translations", len(all_entries), on_epoch=True)"""

    

    
    """def test_step(self, batch, batch_idx):
        source_token_ids, source_mask, ne_tags, target_ids, target_mask = batch

        pred_token_ids = self(
            source_token_ids.to(self.device),
            source_mask.to(self.device),
            training=False
        )

        def filter_valid_ids(token_ids):
            return [id for id in token_ids if 0 <= id < self.tokenizer.vocab_size]

        pred_texts = [
            self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True)
            for ids in pred_token_ids
        ]
        target_texts = [
            self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True)
            for ids in target_ids
        ]

    # Clean texts
        clean_pred_texts = [text.replace('<extra_id_0>', '').strip() for text in pred_texts]
        clean_target_texts = [text.replace('<extra_id_0>', '').strip() for text in target_texts]

    # Compute BLEU score (using sacrebleu)
        bleu_score = sacrebleu.corpus_bleu(clean_pred_texts, [clean_target_texts]).score

    # Compute ROUGE using the rouge_score package
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        for pred, ref in zip(clean_pred_texts, clean_target_texts):
            scores = self.rouge_scorer.score(pred, ref)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        num_samples = len(clean_pred_texts)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples

    # For Meteor, ensure the texts are tokenized into lists of tokens.
        from nltk.tokenize import word_tokenize
        tokenized_pred_texts = [word_tokenize(pred) for pred in clean_pred_texts]
        tokenized_target_texts = [word_tokenize(ref) for ref in clean_target_texts]

        meteor_scores = [
            nltk.translate.meteor_score.meteor_score([ref], pred)
            for pred, ref in zip(tokenized_pred_texts, tokenized_target_texts)
        ]
        meteor_score_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    
        self.log('test_bleu', bleu_score, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('test_rouge1', rouge_scores['rouge1'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('test_rouge2', rouge_scores['rouge2'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('test_rougeL', rouge_scores['rougeL'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('test_meteor', meteor_score_avg, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        return {"bleu": bleu_score, "rouge": rouge_scores, "meteor": meteor_score_avg}"""

       

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
