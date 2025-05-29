import os
import glob
import torch
from transformers import MT5TokenizerFast, AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import argparse

class MT5DatasetPreprocessor:
    def __init__(self, data_dir, source_ext='_en.txt', target_ext='_pt.txt', tokenizer=None, max_len=256):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.train_source_files = sorted(glob.glob(os.path.join(data_dir, "train", f"*{source_ext}")))
        self.train_target_files = sorted(glob.glob(os.path.join(data_dir, "train", f"*{target_ext}")))

        self.val_source_files = sorted(glob.glob(os.path.join(data_dir, "val", f"*{source_ext}")))
        self.val_target_files = sorted(glob.glob(os.path.join(data_dir, "val", f"*{target_ext}")))

        self.test_source_files = sorted(glob.glob(os.path.join(data_dir, "test", f"*{source_ext}")))
        self.test_target_files = sorted(glob.glob(os.path.join(data_dir, "test", f"*{target_ext}")))

        print(f"Found {len(self.train_source_files)} training samples, {len(self.val_source_files)} validation samples and {len(self.test_source_files)} test samples ")

        try:
            self.nlp_spacy = spacy.load("en_ner_bionlp13cg_md")
        except Exception as e:
            raise ValueError("Ensure that the spaCy model 'en_ner_bionlp13cg_md' is installed.") from e

        try:
            # Original HF biomedical NER model
            self.tokenizer_hf = AutoTokenizer.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.model_hf = AutoModelForTokenClassification.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.ner_pipeline = pipeline("ner", model=self.model_hf, tokenizer=self.tokenizer_hf, aggregation_strategy="simple")
            self.ner_pipeline.tokenizer.model_max_length = 512
        except Exception as e:
            raise ValueError("Ensure that the model 'Kushtrim/bert-base-cased-biomedical-ner' is available.") from e

        try:
            # Adding BioBERT model for additional NER tags
            self.tokenizer_biobert = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.model_biobert = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.ner_pipeline_biobert = pipeline("ner", model=self.model_biobert, tokenizer=self.tokenizer_biobert, aggregation_strategy="simple")
            self.ner_pipeline_biobert.tokenizer.model_max_length = 512
        except Exception as e:
            raise ValueError("Ensure that the BioBERT model 'dmis-lab/biobert-base-cased-v1.1' is available.") from e

        # Extend the tag dictionary with tags coming from BioBERT (use appropriate keys/suffixes)
        self.tag_to_idx = {
            'O': 0, 'AMINO_ACID': 1, 'ANATOMICAL_SYSTEM': 2, 'CANCER': 3, 'CELL': 4, 'CELLULAR_COMPONENT': 5,
            'DEVELOPING_ANATOMICAL_STRUCTURE': 6, 'GENE_OR_GENE_PRODUCT': 7, 'IMMATERIAL_ANATOMICAL_ENTITY': 8,
            'MULTI_TISSUE_STRUCTURE': 9, 'ORGAN': 10, 'ORGANISM_SPACY': 11, 'ORGANISM_SUBDIVISION': 12,
            'ORGANISM_SUBSTANCE': 13, 'PATHOLOGICAL_FORMATION': 14, 'SIMPLE_CHEMICAL': 15, 'TISSUE_SPACY': 16,
            'SMALL_MOLECULE': 17, 'GENEPROD': 18, 'SUBCELLULAR': 19, 'CELL_LINE': 20, 'CELL_TYPE': 21,
            'TISSUE_HF': 22, 'ORGANISM_HF': 23, 'DISEASE': 24, 'EXP_ASSAY': 25,
           
            'AMINO_ACID_BIOBERT': 26, 'ANATOMICAL_SYSTEM_BIOBERT': 27, 'CANCER_BIOBERT': 28, # but not using BIOBERT
            
        }
    def align_biobert_tags_with_tokens(self, text, offsets, input_ids):
        """
        Creates a binary tensor aligned with the tokens.
        For each token, it returns 1 if the token overlaps with a BioBERT entity
        (i.e. with label "LABEL_1") and 0 otherwise.
        """
   
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    
    
        bio_results = self.ner_pipeline_biobert(text)
    
   
        bio_entities = []
        for entity in bio_results:
        # Check if the model predicted the entity as LABEL_1
            if entity.get("entity_group") == "LABEL_1":
                bio_entities.append((entity["start"], entity["end"]))
    
   
        aligned_bio_tags = []
        for token, (start, end) in zip(tokens, offsets.tolist()):
            tag = 0  # Default: no entity
            for ent_start, ent_end in bio_entities:
            # If the token's span overlaps with the entity span, mark as 1
                if (start >= ent_start and end <= ent_end) or (start < ent_end and end > ent_start):
                    tag = 1
                    break
            aligned_bio_tags.append(tag)
    
        return torch.tensor(aligned_bio_tags, dtype=torch.long)


    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def combined_ne_tag(self, text):
       
        doc_spacy = self.nlp_spacy(text)
        entities_spacy = [(ent.start_char, ent.end_char, ent.label_) for ent in doc_spacy.ents]
        entities_spacy = [
            (start, end, label if label not in ['TISSUE', 'ORGANISM'] else f"{label}_SPACY")
            for start, end, label in entities_spacy
        ]


        ner_results_batch = self.ner_pipeline([text], batch_size=8)
        entities_hf = []
        for ner_results in ner_results_batch:
            merged_entities = []
            prev_entity = None
            for entity in ner_results:
                if isinstance(entity, dict):
                    ent_start = entity.get('start')
                    ent_end = entity.get('end')
                    ent_label = entity.get('entity_group')
                    if prev_entity and ent_label == prev_entity[2] and ent_start == prev_entity[1]:
                        merged_entities[-1] = (prev_entity[0], ent_end, ent_label)
                        prev_entity = merged_entities[-1]
                    else:
                        merged_entities.append((ent_start, ent_end, ent_label))
                        prev_entity = (ent_start, ent_end, ent_label)
            for ent_start, ent_end, ent_label in merged_entities:
                if ent_label in ['TISSUE', 'ORGANISM']:
                    ent_label += '_HF'
                entities_hf.append((ent_start, ent_end, ent_label))

      
        ner_results_biobert_batch = self.ner_pipeline_biobert([text], batch_size=8)
        entities_biobert = []
        for ner_results in ner_results_biobert_batch:
            merged_entities = []
            prev_entity = None
            for entity in ner_results:
                if isinstance(entity, dict):
                    ent_start = entity.get('start')
                    ent_end = entity.get('end')
                    ent_label = entity.get('entity_group')
                    if prev_entity and ent_label == prev_entity[2] and ent_start == prev_entity[1]:
                        merged_entities[-1] = (prev_entity[0], ent_end, ent_label)
                        prev_entity = merged_entities[-1]
                    else:
                        merged_entities.append((ent_start, ent_end, ent_label))
                        prev_entity = (ent_start, ent_end, ent_label)
            for ent_start, ent_end, ent_label in merged_entities:
                
                ent_label = f"{ent_label}_BIOBERT"
                entities_biobert.append((ent_start, ent_end, ent_label))

        # Combine and sort all entities
        combined_entities = entities_spacy + entities_hf + entities_biobert
        combined_entities.sort(key=lambda x: (x[0], x[1]))
        return combined_entities

    """def align_ne_tags_with_tokens(self, text, entities, offsets, input_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        aligned_tags = []
        for token, (start, end) in zip(tokens, offsets.tolist()):
            tag = 'O'
            for ent_start, ent_end, ent_label in entities:
                if (start >= ent_start and end <= ent_end) or (start < ent_end and end > ent_start):
                    tag = ent_label
                    break
            aligned_tags.append(tag)
        tag_ids = [self.tag_to_idx.get(tag, 0) for tag in aligned_tags]
        return torch.tensor(tag_ids, dtype=torch.long)"""
    
    def align_ne_tags_with_tokens(self, text, entities, offsets, input_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        aligned_tags = []
        for i, (token, (start, end)) in enumerate(zip(tokens, offsets.tolist())):
            token_center = (start + end) / 2.0
            tag = 'O'
            for ent_start, ent_end, ent_label in entities:
                if token_center >= ent_start and token_center <= ent_end:
                    tag = ent_label
                    break
            if i > 0 and not token.startswith("▁") and aligned_tags[i-1] != 'O':
                tag = aligned_tags[i-1]
            aligned_tags.append(tag)
        tag_ids = [self.tag_to_idx.get(tag, 0) for tag in aligned_tags]
        return torch.tensor(tag_ids, dtype=torch.long)

    def process_sample(self, source_text, target_text):
        #source_text = f"translate English to Portuguese: {source_text}"
        tokenized = self.tokenizer(
            source_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_attention_mask=True
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        offsets = tokenized['offset_mapping'].squeeze(0)

   
        entities = self.combined_ne_tag(source_text)
        ne_tags = self.align_ne_tags_with_tokens(source_text, entities, offsets, input_ids)



    # Get binary BioBERT entity tags
        bio_ner_tags = self.align_biobert_tags_with_tokens(source_text, offsets, input_ids)

        tokenized_target = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True
        )
        target_ids = tokenized_target['input_ids'].squeeze(0)
        target_mask = tokenized_target['attention_mask'].squeeze(0)

    # You could then combine these outputs as needed.
        return input_ids, attention_mask, ne_tags, target_ids, target_mask


    def process_all(self, source_files, target_files):
        all_samples = []
        for src_file, tgt_file in zip(source_files, target_files):
            source_text = self.load_file(src_file)
            target_text = self.load_file(tgt_file)
            sample = self.process_sample(source_text, target_text)
            all_samples.append(sample)
        return all_samples

    def print_alignment_table(self, text):
        """
        For debugging and checking alignmemnt: prints a table with each token, its character offsets, and the assigned NE tag.
        It can be removed.

        """
        tokenized = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_len,
            return_tensors="pt", return_offsets_mapping=True, return_attention_mask=True
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        offsets = tokenized['offset_mapping'].squeeze(0)
        entities = self.combined_ne_tag(text)
        ne_tags = self.align_ne_tags_with_tokens(text, entities, offsets, input_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        table_data = []
        for token, offset, tag in zip(tokens, offsets.tolist(), ne_tags.tolist()):
            table_data.append([token, offset[0], offset[1], tag])
        try:
            from tabulate import tabulate
            print(tabulate(table_data, headers=["Token", "Start", "End", "NE Tag"]))
        except ImportError:
            print("Token\tStart\tEnd\tNE Tag")
            for row in table_data:
                print("\t".join(map(str, row)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset and save processed data.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the raw data directory.")
    parser.add_argument("--output_train", type=str, default="./preprocessed_files/train/preprocessed_train.pt", help="File to save preprocessed training data.")
    parser.add_argument("--output_val", type=str, default="./preprocessed_files/val/preprocessed_val.pt", help="File to save preprocessed validation data.")
    parser.add_argument("--output_test", type=str, default="./preprocessed_files/test/preprocessed_test.pt", help="File to save preprocessed test data.")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum token length.")
    parser.add_argument("--debug_text", type=str, help="A text sample to print alignment table for debugging.", default="The BRCA1 gene is crucial for DNA repair and is linked to an increased risk of breast cancer.")
    args = parser.parse_args()

    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base", legacy=False)
    preprocessor = MT5DatasetPreprocessor(data_dir=args.data_dir, tokenizer=tokenizer, max_len=args.max_len)

    # Print alignment table for debugging purposes
    print("Alignment Table for Debug Text:")
    preprocessor.print_alignment_table(args.debug_text)

    print("Processing training samples...")
    train_samples = preprocessor.process_all(preprocessor.train_source_files, preprocessor.train_target_files)
    torch.save(train_samples, args.output_train)
    print(f"Saved {len(train_samples)} training samples.")

    print("Processing validation samples...")
    val_samples = preprocessor.process_all(preprocessor.val_source_files, preprocessor.val_target_files)
    torch.save(val_samples, args.output_val)
    print(f"Saved {len(val_samples)} validation samples.")

    print("Processing test samples...")
    preprocessor.test_source_files = sorted(glob.glob(os.path.join(args.data_dir, "test", "*_en.txt")))
    preprocessor.test_target_files = sorted(glob.glob(os.path.join(args.data_dir, "test", "*_pt.txt")))

    print(f"Found {len(preprocessor.test_source_files)} test English files and {len(preprocessor.test_target_files)} test Portuguese files.")

    test_samples = preprocessor.process_all(preprocessor.test_source_files, preprocessor.test_target_files)
    torch.save(test_samples, args.output_test)
    print(f"Saved {len(test_samples)} test samples.")

    print("Preprocessing complete.")
