import os
import glob
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import T5TokenizerFast, AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

class T5Dataset(Dataset):
    def __init__(self, data_dir, source_ext, target_ext, tokenizer, max_len=512):
        # Collect source and target files
        self.source_files = sorted(glob.glob(os.path.join(data_dir, f"*{source_ext}")))
        self.target_files = sorted(glob.glob(os.path.join(data_dir, f"*{target_ext}")))
        assert len(self.source_files) == len(self.target_files), "Mismatch between source and target files."

        print(f"Found {len(self.source_files)} source files and {len(self.target_files)} target files in {data_dir}")

        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load SpaCy's biomedical NER model
        try:
            self.nlp_spacy = spacy.load("en_ner_bionlp13cg_md")
        except Exception as e:
            raise ValueError("Ensure that the SpaCy model 'en_ner_bionlp13cg_md' is installed.") from e

        # Load Hugging Face's biomedical NER model
        try:
            self.tokenizer_hf = AutoTokenizer.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.model_hf = AutoModelForTokenClassification.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.ner_pipeline = pipeline("ner", model=self.model_hf, tokenizer=self.tokenizer_hf, aggregation_strategy="simple")
        except Exception as e:
            raise ValueError("Ensure that the model 'Kushtrim/bert-base-cased-biomedical-ner' is available.") from e

        # Mapping of NE tags to indices
        self.tag_to_idx = {
            'O': 0,
            # SpaCy NER tags
            'AMINO_ACID': 1,
            'ANATOMICAL_SYSTEM': 2,
            'CANCER': 3,
            'CELL': 4,
            'CELLULAR_COMPONENT': 5,
            'DEVELOPING_ANATOMICAL_STRUCTURE': 6,
            'GENE_OR_GENE_PRODUCT': 7,
            'IMMATERIAL_ANATOMICAL_ENTITY': 8,
            'MULTI_TISSUE_STRUCTURE': 9,
            'ORGAN': 10,
            'ORGANISM_SPACY': 11,
            'ORGANISM_SUBDIVISION': 12,
            'ORGANISM_SUBSTANCE': 13,
            'PATHOLOGICAL_FORMATION': 14,
            'SIMPLE_CHEMICAL': 15,
            'TISSUE_SPACY': 16,
            # Hugging Face NER tags
            'SMALL_MOLECULE': 17,
            'GENEPROD': 18,
            'SUBCELLULAR': 19,
            'CELL_LINE': 20,
            'CELL_TYPE': 21,
            'TISSUE_HF': 22,
            'ORGANISM_HF': 23,
            'DISEASE': 24,
            'EXP_ASSAY': 25,
        }

    def __len__(self):
        return len(self.source_files)

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def combined_ne_tag(self, text):
        # Process the text with SpaCy
        doc_spacy = self.nlp_spacy(text)
        entities_spacy = [(ent.start_char, ent.end_char, ent.label_) for ent in doc_spacy.ents]
        # Adjust labels if necessary
        entities_spacy = [
            (start, end, label if label not in ['TISSUE', 'ORGANISM'] else f"{label}_SPACY")
            for start, end, label in entities_spacy
        ]

        # Process the text with Hugging Face model
        ner_results = self.ner_pipeline(text)
        entities_hf = []
        for entity in ner_results:
            ent_start = entity['start']
            ent_end = entity['end']
            ent_label = entity['entity_group']
            # Adjust labels if necessary
            if ent_label in ['TISSUE', 'ORGANISM']:
                ent_label += '_HF'
            entities_hf.append((ent_start, ent_end, ent_label))

        # Combine entities
        combined_entities = entities_spacy + entities_hf

        # Resolve overlapping entities
        combined_entities = self.resolve_overlaps(combined_entities, entities_hf)

        return combined_entities

    def resolve_overlaps(self, entities, entities_hf):
        # Sort entities by start_char and end_char
        entities = sorted(entities, key=lambda x: (x[0], x[1]))
        resolved_entities = []
        for ent in entities:
            if not resolved_entities:
                resolved_entities.append(ent)
            else:
                last_ent = resolved_entities[-1]
                # Check for overlap
                if ent[0] < last_ent[1]:
                    # Decide which entity to keep
                    # For this example, we give priority to Hugging Face entities
                    if ent in entities_hf:
                        resolved_entities[-1] = ent
                else:
                    resolved_entities.append(ent)
        return resolved_entities

    def preprocess(self, text):
        # Get combined entities
        entities = self.combined_ne_tag(text)
        # Tokenize the text with offset mappings
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_attention_mask=True
        )

        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        offsets = tokenized_text['offset_mapping'].squeeze(0)

        # Align NE tags with tokens
        aligned_ne_tags = self.align_ne_tags_with_tokens(text, entities, offsets, input_ids)
        return input_ids, attention_mask, aligned_ne_tags

    def align_ne_tags_with_tokens(self, text, entities, offsets, input_ids):
        aligned_ne_tags = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        for idx, (token, (start, end)) in enumerate(zip(tokens, offsets.tolist())):
            if start == end:
                # Special tokens like <pad> have start == end == 0
                aligned_ne_tags.append('O')
            else:
                tag = 'O'  # Default to 'O', which means outside of any named entity

                # Iterate over each entity to determine if this token falls within any entity span
                for ent_start, ent_end, ent_label in entities:
                    # The condition to tag a token is:
                    # 1. If the token is fully within an entity.
                    # 2. If the token partially overlaps with an entity.
                    if (start >= ent_start and end <= ent_end) or (start < ent_end and end > ent_start):
                        # Assign the entity label to this token if it overlaps with the entity
                        tag = ent_label
                        break  # Stop after finding the first match

                aligned_ne_tags.append(tag)

        # Convert NE tags to indices using the tag_to_idx mapping
        aligned_ne_tag_ids = torch.tensor(
            [self.tag_to_idx.get(tag, 0) for tag in aligned_ne_tags],
            dtype=torch.long
        )

        # Pad or truncate the NE tag tensor to ensure it matches max_len
        if aligned_ne_tag_ids.size(0) < self.max_len:
            pad_length = self.max_len - aligned_ne_tag_ids.size(0)
            aligned_ne_tag_ids = torch.cat([aligned_ne_tag_ids, torch.zeros(pad_length, dtype=torch.long)])
        else:
            aligned_ne_tag_ids = aligned_ne_tag_ids[:self.max_len]

        return aligned_ne_tag_ids

    def preprocess_target(self, text):
        # Tokenize the target text without NE tagging
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        return input_ids, attention_mask

    def __getitem__(self, idx):
        source_text = self.load_file(self.source_files[idx])
        target_text = self.load_file(self.target_files[idx])

        source_input_ids, source_attention_mask, source_ne_tags = self.preprocess(source_text)
        target_input_ids, target_attention_mask = self.preprocess_target(target_text)

        return source_input_ids, source_attention_mask, source_ne_tags, target_input_ids, target_attention_mask

def create_dataloaders(data_dir, tokenizer, batch_size, num_workers=4):
    train_dataset = T5Dataset(
        data_dir=os.path.join(data_dir, 'train'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )
    val_dataset = T5Dataset(
        data_dir=os.path.join(data_dir, 'val'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )
    test_dataset = T5Dataset(
        data_dir=os.path.join(data_dir, 'test'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the file paths and dataset directory structure.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the file paths and dataset directory structure.")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Please check the file paths and dataset directory structure.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    # Set data directory and batch size
    data_dir = "./data"  # Adjust this path to your data directory
    batch_size = 1 # Adjust as per your configuration

    # Create DataLoaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(data_dir, tokenizer, batch_size)








""" import os
import glob
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import T5TokenizerFast, AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import logging

logging.basicConfig(level=logging.INFO)

class T5Dataset(Dataset):
    def __init__(self, data_dir, source_ext, target_ext, tokenizer, max_len=512):
        # Collect source and target files
        self.source_files = sorted(glob.glob(os.path.join(data_dir, f"*{source_ext}")))
        self.target_files = sorted(glob.glob(os.path.join(data_dir, f"*{target_ext}")))
        assert len(self.source_files) == len(self.target_files), "Mismatch between source and target files."

        logging.info(f"Found {len(self.source_files)} source files and {len(self.target_files)} target files in {data_dir}")

        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load SpaCy's biomedical NER model
        try:
            self.nlp_spacy = spacy.load("en_ner_bionlp13cg_md")
        except Exception as e:
            raise ValueError("Ensure that the SpaCy model 'en_ner_bionlp13cg_md' is installed.") from e

        # Load Hugging Face's biomedical NER model
        try:
            self.tokenizer_hf = AutoTokenizer.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.model_hf = AutoModelForTokenClassification.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.ner_pipeline = pipeline("ner", model=self.model_hf, tokenizer=self.tokenizer_hf, aggregation_strategy="simple")
        except Exception as e:
            raise ValueError("Ensure that the model 'Kushtrim/bert-base-cased-biomedical-ner' is available.") from e

        # Mapping of NE tags to indices
        self.tag_to_idx = self.create_tag_to_idx()

    def create_tag_to_idx(self):
        return {
            'O': 0,
            # SpaCy NER tags
            'AMINO_ACID': 1,
            'ANATOMICAL_SYSTEM': 2,
            'CANCER': 3,
            'CELL': 4,
            'CELLULAR_COMPONENT': 5,
            'DEVELOPING_ANATOMICAL_STRUCTURE': 6,
            'GENE_OR_GENE_PRODUCT': 7,
            'IMMATERIAL_ANATOMICAL_ENTITY': 8,
            'MULTI_TISSUE_STRUCTURE': 9,
            'ORGAN': 10,
            'ORGANISM_SPACY': 11,
            'ORGANISM_SUBDIVISION': 12,
            'ORGANISM_SUBSTANCE': 13,
            'PATHOLOGICAL_FORMATION': 14,
            'SIMPLE_CHEMICAL': 15,
            'TISSUE_SPACY': 16,
            # Hugging Face NER tags
            'SMALL_MOLECULE': 17,
            'GENEPROD': 18,
            'SUBCELLULAR': 19,
            'CELL_LINE': 20,
            'CELL_TYPE': 21,
            'TISSUE_HF': 22,
            'ORGANISM_HF': 23,
            'DISEASE': 24,
            'EXP_ASSAY': 25,
        }

    def __len__(self):
        return len(self.source_files)

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def combined_ne_tag(self, text):
        # Process the text with SpaCy
        doc_spacy = self.nlp_spacy(text)
        entities_spacy = [(ent.start_char, ent.end_char, ent.label_) for ent in doc_spacy.ents]
        # Adjust labels if necessary
        entities_spacy = [
            (start, end, label if label not in ['TISSUE', 'ORGANISM'] else f"{label}_SPACY")
            for start, end, label in entities_spacy
        ]

        # Process the text with Hugging Face model
        ner_results = self.ner_pipeline(text)
        entities_hf = []
        for entity in ner_results:
            ent_start = entity['start']
            ent_end = entity['end']
            ent_label = entity['entity_group']
            # Adjust labels if necessary
            if ent_label in ['TISSUE', 'ORGANISM']:
                ent_label += '_HF'
            entities_hf.append((ent_start, ent_end, ent_label))

        # Combine entities
        combined_entities = entities_spacy + entities_hf

        # Resolve overlapping entities
        combined_entities = self.resolve_overlaps(combined_entities, entities_hf)

        return combined_entities

    def resolve_overlaps(self, entities, entities_hf):
        # Sort entities by start_char and end_char
        entities = sorted(entities, key=lambda x: (x[0], x[1]))
        resolved_entities = []
        for ent in entities:
            if not resolved_entities:
                resolved_entities.append(ent)
            else:
                last_ent = resolved_entities[-1]
                # Check for overlap
                if ent[0] < last_ent[1]:
                    # Decide which entity to keep
                    # For this example, we give priority to Hugging Face entities
                    if ent in entities_hf:
                        resolved_entities[-1] = ent
                else:
                    resolved_entities.append(ent)
        return resolved_entities

    def preprocess(self, text):
        # Get combined entities
        entities = self.combined_ne_tag(text)
        # Tokenize the text with offset mappings
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_attention_mask=True
        )

        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        offsets = tokenized_text['offset_mapping'].squeeze(0)

        # Align NE tags with tokens
        aligned_ne_tags = self.align_ne_tags_with_tokens(text, entities, offsets, input_ids)
        return input_ids, attention_mask, aligned_ne_tags

    def align_ne_tags_with_tokens(self, text, entities, offsets, input_ids):
        aligned_ne_tags = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        for idx, (token, (start, end)) in enumerate(zip(tokens, offsets.tolist())):
            if start == end:
                # Special tokens like <pad> have start == end == 0
                aligned_ne_tags.append('O')
            else:
                tag = 'O'  # Default to 'O', which means outside of any named entity

                # Iterate over each entity to determine if this token falls within any entity span
                for ent_start, ent_end, ent_label in entities:
                    # The condition to tag a token is:
                    # 1. If the token is fully within an entity.
                    # 2. If the token partially overlaps with an entity.
                    if (start >= ent_start and end <= ent_end) or (start < ent_end and end > ent_start):
                        # Assign the entity label to this token if it overlaps with the entity
                        tag = ent_label
                        break  # Stop after finding the first match

                aligned_ne_tags.append(tag)

        # Convert NE tags to indices using the tag_to_idx mapping
        aligned_ne_tag_ids = torch.tensor(
            [self.tag_to_idx.get(tag, 0) for tag in aligned_ne_tags],
            dtype=torch.long
        )

        # Pad or truncate the NE tag tensor to ensure it matches max_len
        aligned_ne_tag_ids = self.pad_or_truncate_ne_tags(aligned_ne_tag_ids)

        return aligned_ne_tag_ids

    def pad_or_truncate_ne_tags(self, aligned_ne_tag_ids):
        if aligned_ne_tag_ids.size(0) < self.max_len:
            pad_length = self.max_len - aligned_ne_tag_ids.size(0)
            aligned_ne_tag_ids = torch.cat([aligned_ne_tag_ids, torch.zeros(pad_length, dtype=torch.long)])
        else:
            aligned_ne_tag_ids = aligned_ne_tag_ids[:self.max_len]
        return aligned_ne_tag_ids

    def preprocess_target(self, text):
        # Tokenize the target text without NE tagging
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        return input_ids, attention_mask

    def __getitem__(self, idx):
        source_text = self.load_file(self.source_files[idx])
        target_text = self.load_file(self.target_files[idx
"""

#to try just in case



