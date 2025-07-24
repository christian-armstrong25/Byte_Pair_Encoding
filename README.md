# Byte Pair Encoding (BPE) Tokenizer

This project demonstrates a simple, from-scratch implementation of Byte Pair Encoding (BPE) tokenization, inspired by GPT-2. It shows how to train a BPE tokenizer on a text corpus, handling all languages by working at the byte level.

## What This Project Does

- **Trains a BPE tokenizer**: Builds a vocabulary and merge rules from a text file.
- **Handles special tokens**: Supports custom tokens like `<|endoftext|>`.
- **Pre-tokenizes text**: Uses a regex (like GPT-2) to split text into meaningful units before encoding.

## Key Functions

- `count_pre_tokens(chunk, r_split)`: Splits text into pre-tokens using regex and counts their byte-encoded forms.
- `train(input_path, vocab_size, special_tokens)`: Trains the BPE tokenizer, returning the vocabulary and merge rules.

## How to Use

Run the script to train a tokenizer on `data/sample.txt`:

```bash
python blank.py
```

## Why This Matters

- Shows how modern tokenizers work under the hood
- Handles any language or character set (byte-level)
- Demonstrates core NLP preprocessing techniques

---

See `BPE.md` for algorithm details and references.
