"""
Data Loading Utilities
======================
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class WikitextDataset(Dataset):
    def __init__(self, encodings, seq_length):
        # encodings.input_ids is [1, total_len]
        self.input_ids = encodings.input_ids[0]
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.input_ids) - 1) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        input_ids = self.input_ids[start_idx:end_idx]
        target_ids = self.input_ids[start_idx+1:end_idx+1]
        # Return as LongTensor
        return input_ids.long(), target_ids.long()

class CharDataset(Dataset):
    """Character-level language modeling dataset."""
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.data) - 1) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        x = self.data[start:end].long()
        y = self.data[start + 1:end + 1].long()
        return x, y


def get_wikitext_char_dataloader(seq_length=256, batch_size=32):
    """
    Load Wikitext-2 as character-level data.
    Returns: (train_loader, val_loader, vocab_size, itos)
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Build vocabulary from training data
    train_text = "\n".join(dataset["train"]["text"])
    val_text = "\n".join(dataset["validation"]["text"])
    test_text = "\n".join(dataset["test"]["text"])

    chars = sorted(set(train_text + val_text + test_text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)

    def encode(text):
        return torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)

    train_ids = encode(train_text)
    val_ids = encode(val_text)

    print(f"Char vocab size: {vocab_size}")
    print(f"Train chars: {len(train_ids):,}")
    print(f"Val chars: {len(val_ids):,}")

    train_ds = CharDataset(train_ids, seq_length)
    val_ds = CharDataset(val_ids, seq_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2,
    )

    return train_loader, val_loader, vocab_size, itos


def get_wikitext_dataloader(seq_length=512, batch_size=32, tokenizer_name="gpt2"):
    """
    Load Wikitext-2, tokenize, and return DataLoaders for train/validation.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="pt", truncation=False, padding=False)

    # Tokenize the entire dataset into one long stream
    # Note: This is a simplified approach. For production, we usually chunk properly.
    # Joining all text with EOS token
    def group_texts(split_name):
        texts = dataset[split_name]["text"]
        full_text = "".join([t + tokenizer.eos_token for t in texts if t.strip()])
        tokenized = tokenizer(full_text, return_tensors="pt", max_length=len(full_text), truncation=False)
        return tokenized

    print(f"Tokenizing {tokenizer_name} on Wikitext-2...")
    # Hack to prevent warning
    tokenizer.model_max_length = 100_000_000
    
    train_encodings = group_texts("train")
    val_encodings = group_texts("validation")
    
    print(f"Train tokens: {train_encodings.input_ids.size(1)}")
    print(f"Val tokens: {val_encodings.input_ids.size(1)}")

    train_dataset = WikitextDataset(train_encodings, seq_length)
    val_dataset = WikitextDataset(val_encodings, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, tokenizer.vocab_size

