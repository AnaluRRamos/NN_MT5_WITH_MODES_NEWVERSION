# test_preprocess.py

import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import T5TokenizerFast
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from termcolor import colored

from preprocess import T5Dataset, create_dataloaders

def print_entities(text, entities):
    """Print the original text with the identified entities."""
    print("\nOriginal Text:")
    print(text)
    print("\nEntities Identified:")
    for start, end, label in entities:
        print(f"  Entity: '{text[start:end]}' - Label: {label}")

def highlight_entities(text, entities):
    """Print the text with entities highlighted for easy visualization."""
    highlighted_text = ""
    last_index = 0
    for start, end, label in sorted(entities, key=lambda x: x[0]):
        highlighted_text += text[last_index:start]
        highlighted_text += colored(text[start:end], 'yellow', attrs=['bold'])
        last_index = end
    highlighted_text += text[last_index:]
    print("\nHighlighted Text with Entities:")
    print(highlighted_text)

def print_token_tag_alignment(tokens, tags):
    """Display token-NE tag alignment in a table format."""
    data = {'Token': tokens, 'NE Tag': tags}
    df = pd.DataFrame(data)
    print("\nToken-Tag Alignment Table:")
    print(df)

def plot_named_entity_distribution(all_ne_tags):
    """Plot the distribution of named entity tags."""
    tag_counts = Counter(all_ne_tags)
    labels, counts = zip(*tag_counts.items())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.xlabel('Tag Index')
    plt.ylabel('Frequency')
    plt.title('Named Entity Tag Distribution')
    plt.show()

if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    # Set data directory and batch size
    data_dir = "/Users/mac/Desktop/NN_MT_T5_WITH_MODES/data"  # Replace with your actual data directory
    batch_size = 2  # Use a small batch size for testing

    # Create the full training dataset
    train_dataset = T5Dataset(
        data_dir=os.path.join(data_dir, 'train'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )

    # Create a subset of the dataset with only 5 samples
    subset_indices = list(range(5))  # Indices of the first 5 samples
    train_subset = Subset(train_dataset, subset_indices)

    # Create a DataLoader for the subset
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)

    # Iterate over the DataLoader to test preprocessing
    all_ne_tags = []
    print("Testing the preprocessing with a subset of 5 samples:")
    for batch in train_dataloader:
        (source_input_ids, source_attention_mask, source_ne_tags,
         target_input_ids, target_attention_mask) = batch

        # Iterate through each sample in the batch
        for i in range(source_input_ids.size(0)):
            # Load the original source text
            source_text = train_dataset.load_file(train_dataset.source_files[i])

            # Load and print entities
            entities = train_dataset.combined_ne_tag(source_text)
            print_entities(source_text, entities)

            # Highlight entities in the original text
            highlight_entities(source_text, entities)

            # Convert tokens and print token-tag alignment
            tokens = tokenizer.convert_ids_to_tokens(source_input_ids[i])
            tags = source_ne_tags[i].tolist()
            print_token_tag_alignment(tokens, tags)

            # Collect NE tags for plotting later
            all_ne_tags.extend(tags)

            # Print the shapes of the tensors
            print("\nTensor Shapes:")
            print("Source Input IDs shape:", source_input_ids.shape)
            print("Source Attention Mask shape:", source_attention_mask.shape)
            print("Source NE Tags shape:", source_ne_tags.shape)
            print("Target Input IDs shape:", target_input_ids.shape)
            print("Target Attention Mask shape:", target_attention_mask.shape)

            # Optionally, print the actual data for tokens and target tokens
            print(f"\nSample {i+1}:")
            print("Source Tokens:", tokens)
            print("Source NE Tags:", tags)
            print("Target Tokens:", tokenizer.convert_ids_to_tokens(target_input_ids[i]))

        # Break after one batch to limit output
        break

    # Plot the named entity tag distribution for all samples in the subset
    plot_named_entity_distribution(all_ne_tags)












"""import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import T5TokenizerFast

from preprocess import T5Dataset, create_dataloaders

if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    # Set data directory and batch size
    data_dir = "/Users/mac/Desktop/NN_MT_T5_WITH_MODES/data"  # Replace with your actual data directory
    batch_size = 2  # Use a small batch size for testing

    # Create the full training dataset
    train_dataset = T5Dataset(
        data_dir=os.path.join(data_dir, 'train'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )

    # Create a subset of the dataset with only 5 samples
    subset_indices = list(range(5))  # Indices of the first 5 samples
    train_subset = Subset(train_dataset, subset_indices)

    # Create a DataLoader for the subset
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)

    # Iterate over the DataLoader to test preprocessing
    print("Testing the preprocessing with a subset of 5 samples:")
    for batch in train_dataloader:
        (source_input_ids, source_attention_mask, source_ne_tags,
         target_input_ids, target_attention_mask, target_ne_tags) = batch

        # Print the shapes of the tensors
        print("Source Input IDs shape:", source_input_ids.shape)
        print("Source Attention Mask shape:", source_attention_mask.shape)
        print("Source NE Tags shape:", source_ne_tags.shape)
        print("Target Input IDs shape:", target_input_ids.shape)
        print("Target Attention Mask shape:", target_attention_mask.shape)
        print("Target NE Tags shape:", target_ne_tags.shape)

        # Optionally, print the actual data
        for i in range(source_input_ids.size(0)):
            print(f"\nSample {i+1}:")
            print("Source Tokens:", tokenizer.convert_ids_to_tokens(source_input_ids[i]))
            print("Source NE Tags:", source_ne_tags[i].tolist())
            print("Target Tokens:", tokenizer.convert_ids_to_tokens(target_input_ids[i]))
            print("Target NE Tags:", target_ne_tags[i].tolist())

        # Break after one batch to limit output
        break"""
