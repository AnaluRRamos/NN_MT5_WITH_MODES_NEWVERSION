import torch
import pytorch_lightning as pl
from torch import nn
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
import sacrebleu
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk 
from src.mode_config import ModeConfig
from src.loss_functions import entity_aware_loss, ner_auxiliary_loss, placeholder_loss, gradual_weight_leaky
import logging
import os
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.loggers import CometLogger 
from comet import download_model, load_from_checkpoint


from torchmetrics.text.bleu import BLEUScore

#from datasets import load_metric
import evaluate
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")


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
    def __init__(self, tokenizer, train_dataloader, val_dataloader, test_dataloader, learning_rate, target_max_length=600, mode=0, num_ne_tags=29):
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
        self.all_bleu_scores = []
        
        logging.info(f"Initialized MT5FineTuner with mode={mode}, learning_rate={learning_rate}, target_max_length={target_max_length}")
        logging.info(f"NE Tag Embedding Shape: {self.ne_tag_embedding.weight.shape}")

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # Initialize BLEU scorer using sacrebleu
        #self.metric_bleu = sacrebleu.metrics.BLEU() 
        self.metric_bleu = None  # sacrebleu is used at the sentence level now

        # COMET Metric 
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        self.comet_model = load_from_checkpoint(comet_model_path)
        self.comet_model.eval()



    
    def forward(self, source_token_ids, source_mask, target_token_ids=None, target_mask=None, ne_tag_mask=None, training=True):
        # if we are in the training we calculate the loss 
        if training:

            labels = target_token_ids.clone()
            labels[target_token_ids == self.tokenizer.pad_token_id] = -100 # -100 ignore padding

            if self.mode == 1 and ne_tag_mask is not None:
            #  [ADDED] Inject NE tag embeddings into encoder input during training
                input_embeds = self.model.encoder.embed_tokens(source_token_ids)
                ne_embeds = self.ne_tag_embedding(ne_tag_mask)
                input_embeds = input_embeds + ne_embeds

                outputs = self.model(
                    inputs_embeds=input_embeds,  
                    attention_mask=source_mask,
                    labels=labels,
                    return_dict=True
                )
            else:
                outputs = self.model(
                    input_ids=source_token_ids,
                    attention_mask=source_mask,
                    labels=labels,
                    return_dict=True
                )

            lm_logits = outputs.logits

            if self.mode == 0:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            elif self.mode == 1:
                warmup_epochs = getattr(ModeConfig, "WARMUP_EPOCHS", 5)
                scaling = min(1.0, self.current_epoch / warmup_epochs)
                current_weight_factor = gradual_weight_leaky(scaling, negative_slope=0.1, target_weight=ModeConfig.MODE_1_WEIGHT)

                if self.current_epoch % 1 == 0 and self.global_step % 10 == 0:
                    print(f"[Epoch {self.current_epoch}] Weight factor for NE loss: {current_weight_factor:.4f}")

                label_smoothing_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
                lm_logits_flat = lm_logits.view(-1, lm_logits.size(-1)) #This flattens the logits from shape [batch_size, seq_len, vocab_size] → [batch_size * seq_len, vocab_size].
#It’s necessary because PyTorch loss functions expect 2D inputs: (N, C) where N = number of examples, and C = number of classes.
                labels_flat = labels.view(-1) #Flattens [batch_size, seq_len] into [batch_size * seq_len] —  ground truth.
                non_ner_mask = (ne_tag_mask.view(-1) == 0) #Give me only the tokens where ne_tag_mask == 0 — i.e., no NE tags
                non_ner_loss = label_smoothing_loss_fn(lm_logits_flat[non_ner_mask], labels_flat[non_ner_mask]) #Applies the label smoothing loss only to non-entity tokens (filtered using the mask).

                ner_loss = entity_aware_loss(lm_logits, labels, ne_tag_mask, weight_factor=current_weight_factor)

                loss = non_ner_loss + ner_loss

            elif self.mode == 2: # for future create different modes to test 
                base_loss = nn.CrossEntropyLoss(ignore_index=-100)(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = placeholder_loss(base_loss, ne_tag_mask) * ModeConfig.MODE_2_WEIGHT

            else:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)) * ModeConfig.MODE_0_WEIGHT

            return loss

        else:
            if self.mode == 1 and ne_tag_mask is not None:
            
                input_embeds = self.model.encoder.embed_tokens(source_token_ids)
                ne_embeds = self.ne_tag_embedding(ne_tag_mask)
                input_embeds = input_embeds + ne_embeds

                generated_token_ids = self.model.generate(
                    inputs_embeds=input_embeds,  
                    attention_mask=source_mask,
                    max_length=self.target_max_length,
                    num_beams=8,
                    early_stopping=True
                )
            else:
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
    
       # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_token_ids, source_mask, source_ne_tags, target_token_ids, target_mask = batch
        
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

        # Decode source input for COMET
        source_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in source_token_ids]

    # Prepare input format for COMET
        #comet_data = [
            #{"src": src, "mt": pred, "ref": ref}
            #for src, pred, ref in zip(source_texts, clean_pred_texts, clean_target_texts)
        #with torch.no_grad():
            #comet_scores = self.comet_model.predict(comet_data, batch_size=8, gpus=1)  # change to gpus=0 for CPU
        #avg_comet_score = sum(comet_scores["scores"]) / len(comet_scores["scores"])

        
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
        
        #meteor_scores = [nltk.translate.meteor_score.meteor_score([ref], pred) for pred, ref in zip(tokenized_pred_texts, tokenized_target_texts)]
        #meteor_score_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

        #chrf_score = chrf_metric.compute(predictions=clean_pred_texts, references=[[ref] for ref in clean_target_texts])["score"]
        #self.log('val_chrf', chrf_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        if batch_idx < 1:  # For example, print predictions from the first batch only
            print("Validation Examples:")
            for pred, ref in zip(pred_texts, target_texts):
                print("Prediction:", pred)
                print("Reference: ", ref)
                print("-" * 50)

        
        #self.log('val_bleu', bleu_score, prog_bar=True, on_epoch=True, sync_dist=True)
        #self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_bleu', bleu_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rouge1', rouge_scores['rouge1'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rouge2', rouge_scores['rouge2'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_rougeL', rouge_scores['rougeL'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('val_meteor', meteor_score_avg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('val_comet', avg_comet_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)



            
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

    # Clean texts
        clean_pred_texts = [text.replace('<extra_id_0>', '').strip() for text in pred_texts]
        clean_target_texts = [text.replace('<extra_id_0>', '').strip() for text in target_texts]

    # Compute BLEU score
        bleu_scores = [
            sacrebleu.sentence_bleu(pred, [ref]).score
            for pred, ref in zip(clean_pred_texts, clean_target_texts)
        ]
    
        self.all_bleu_scores.extend(bleu_scores)

    # ROUGE
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        for pred, ref in zip(clean_pred_texts, clean_target_texts):
            scores = self.rouge_scorer.score(pred, ref)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        num_samples = len(clean_pred_texts)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples

    # METEOR
        tokenized_pred_texts = [word_tokenize(pred) for pred in clean_pred_texts]
        tokenized_target_texts = [word_tokenize(ref) for ref in clean_target_texts]
        meteor_scores = [
            nltk.translate.meteor_score.meteor_score([ref], pred)
            for pred, ref in zip(tokenized_pred_texts, tokenized_target_texts)
        ]
        meteor_score_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # Store results for inspection
        for pred, ref, bleu in zip(clean_pred_texts, clean_target_texts, bleu_scores):
            self.all_test_preds.append({"prediction": pred, "reference": ref, "bleu": bleu})

    # CHrf
        chrf_score = chrf_metric.compute(predictions=clean_pred_texts, references=[[ref] for ref in clean_target_texts])["score"]
        self.log('test_chrf', chrf_score, prog_bar=True, on_epoch=True, sync_dist=True)
    
    # COMET 

        # Decode source input for COMET
        source_texts = [
            self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True)
            for ids in source_token_ids
        ]

    # Prepare COMET input
        comet_data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(source_texts, clean_pred_texts, clean_target_texts)
        ]

    # Compute COMET scores
        with torch.no_grad():
            comet_scores = self.comet_model.predict(comet_data, batch_size=8, gpus=1)  # Or gpus=0 if no GPU
        avg_comet_score = sum(comet_scores["scores"]) / len(comet_scores["scores"])

    # Log COMET
        self.log('test_comet', avg_comet_score, prog_bar=True, on_epoch=True, sync_dist=True)

    # and other Log metrics
        self.log('test_bleu', sum(bleu_scores) / len(bleu_scores), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_rouge1', rouge_scores['rouge1'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_rouge2', rouge_scores['rouge2'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_rougeL', rouge_scores['rougeL'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_meteor', meteor_score_avg, prog_bar=True, on_epoch=True, sync_dist=True)

    #  Save BLEU scores on the last batch only (to avoid writing many times)
        #if batch_idx + 1 == self.trainer.num_test_batches[0]:
            #with open(f"bleu_scores_mode_{self.mode}.txt", "w", encoding="utf-8") as f:
                #for score in self.all_bleu_scores:
                    #f.write(f"{score}\n")

        #return {
            #"pred_texts": clean_pred_texts,
            #"bleu": bleu_scores,
            #"rouge": rouge_scores,
            #"meteor": meteor_score_avg
        #}
    def on_test_epoch_end(self):
    # Sort predictions by BLEU score
        sorted_preds = sorted(self.all_test_preds, key=lambda x: x["bleu"], reverse=True)

    # Get top 10 and bottom 10 translations
        #top_10 = sorted_preds[:10]
        #bottom_10 = sorted_preds[-10:]
        top_1800 = sorted_preds[:1800]

        def save_translations(filename, translations):
            with open(filename, "w", encoding="utf-8") as f:
                for idx, entry in enumerate(translations, start=1):
                    f.write(f"Rank: {idx}\n")
                    f.write(f"BLEU Score: {entry['bleu']:.4f}\n")
                    f.write(f"Prediction: {entry['prediction']}\n")
                    f.write(f"Reference: {entry['reference']}\n\n")

    # Save files
        #save_translations(f"top_10_test_translations_mode_{self.mode}.txt", top_10)
        #save_translations(f"worst_10_test_translations_mode_{self.mode}.txt", bottom_10)
        save_translations(f"top_1800_test_translations_may__mode_{self.mode}.txt", top_1800)

    # Log average BLEU score
        avg_bleu = sum([entry["bleu"] for entry in self.all_test_preds]) / len(self.all_test_preds) if self.all_test_preds else 0.0
        self.log('test_bleu_avg', avg_bleu, prog_bar=True, on_epoch=True, sync_dist=True)




    
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
        bleu_scores = [
            sacrebleu.sentence_bleu(pred, [ref]).score
            for pred, ref in zip(clean_pred_texts, clean_target_texts)
        ]

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

    # Compute METEOR score
        from nltk.tokenize import word_tokenize
        tokenized_pred_texts = [word_tokenize(pred) for pred in clean_pred_texts]
        tokenized_target_texts = [word_tokenize(ref) for ref in clean_target_texts]

        meteor_scores = [
            nltk.translate.meteor_score.meteor_score([ref], pred)
            for pred, ref in zip(tokenized_pred_texts, tokenized_target_texts)
        ]
        meteor_score_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # Store results for ranking later
        for pred, ref, bleu in zip(clean_pred_texts, clean_target_texts, bleu_scores):
            self.all_test_preds.append({"prediction": pred, "reference": ref, "bleu": bleu})

    # Log test metrics
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    # Calculate total training steps: number of batches in training dataloader times max epochs
        total_steps = len(self._train_dataloader) * self.trainer.max_epochs
        warmup_steps = int(0.1 * total_steps)  # Use 10% of steps as warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update scheduler every training step
                'frequency': 1
            }
        }"""



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
