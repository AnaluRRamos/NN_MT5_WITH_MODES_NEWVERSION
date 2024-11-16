import sacrebleu
import torch
import logging
from transformers import T5TokenizerFast
from datasets import load_metric

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load metrics from the Hugging Face `datasets` library
rouge = load_metric("rouge")
meteor = load_metric("meteor")

def evaluate_model(model, dataloader):
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else T5TokenizerFast.from_pretrained('t5-base')
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device)
    model.eval()

    predictions, references = [], []

    for idx, batch in enumerate(dataloader):
        source_ids, source_mask, target_ids, _, _, _ = batch
        source_ids, source_mask, target_ids = (x.to(model.device) for x in [source_ids, source_mask, target_ids])

        with torch.no_grad():
            generated_ids = model.generate(input_ids=source_ids, attention_mask=source_mask, max_length=model.target_max_length)

        # Decode predictions and references
        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        ref_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]

        predictions.extend(pred_texts)
        references.extend(ref_texts)

        # Log evaluation progress
        if idx % 10 == 0:
            logging.info(f"Evaluated batch {idx + 1}/{len(dataloader)}")

    # Calculate BLEU score
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    logging.info(f"BLEU Score: {bleu_score}")

    # Calculate ROUGE score
    rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    rouge_score = rouge_result['rougeL'].mid.fmeasure
    logging.info(f"ROUGE-L F1 Score: {rouge_score}")

    # Calculate METEOR score
    meteor_result = meteor.compute(predictions=predictions, references=references)
    meteor_score = meteor_result['meteor']
    logging.info(f"METEOR Score: {meteor_score}")

    # Display sample prediction and reference for qualitative evaluation
    if len(predictions) > 0:
        logging.info(f"Sample Prediction: {predictions[0]}")
        logging.info(f"Reference: {references[0]}")

    # Return all metrics as a dictionary
    return {
        "BLEU": bleu_score,
        "ROUGE-L": rouge_score,
        "METEOR": meteor_score
    }

