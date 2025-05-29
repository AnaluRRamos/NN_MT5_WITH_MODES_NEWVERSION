import torch
from transformers import MT5TokenizerFast, MT5ForConditionalGeneration
import logging
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from src.utils import MT5DatasetPreprocessed
from src.config import Config  

# Download NLTK data (only once)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, dataloader, tokenizer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions, references = [], []

    for batch in dataloader:
        source_ids, source_mask, ne_tags, target_ids, target_mask = [b.to(device) for b in batch]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=Config.TARGET_MAX_LENGTH, 
                num_beams=8,
                early_stopping=True
            )

        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        ref_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]

        predictions.extend(pred_texts)
        references.extend(ref_texts)

    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {metric: 0.0 for metric in ['rouge1', 'rouge2', 'rougeL']}

    for pred, ref in zip(predictions, references):
        scores = rouge_scorer_obj.score(pred, ref)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure

    num_samples = len(predictions)
    for key in rouge_scores:
        rouge_scores[key] /= num_samples

    meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(predictions, references)]
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    logging.info(f"BLEU: {bleu_score:.2f}, ROUGE: {rouge_scores}, METEOR: {meteor_avg:.2f}")
    return {"bleu": bleu_score, "rouge": rouge_scores, "meteor": meteor_avg}

if __name__ == "__main__":
    model_dir = "./models/mt5_finetuned"  # Replace with your model directory
    tokenizer = MT5TokenizerFast.from_pretrained(model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)

    test_dataset = MT5DatasetPreprocessed("preprocessed_val.pt")
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    metrics = evaluate_model(model, test_dataloader, tokenizer)
