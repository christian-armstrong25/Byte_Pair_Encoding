# Task

We are going to implement Byte-Pair Encoding GPT-2 style!

[wiki](https://en.wikipedia.org/wiki/Byte-pair_encoding#Example)

### Scope

Train a BPE tokenize on the TinyStories dataset (see [data/sample.txt])

### Algorithm
- My corpus looks like:
    "hug pug bun hug pug bun pun ..."
- Get my subwords:
    ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
- Run BPE:
    Step 1: 
        - Vocabulary: ["b", "g", "h", "n", "p", "s", "u"]
        - Pre-tokens: ("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
    Step 2. 
        - Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
        - Pre-tokens: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
    ...

Remember to break ties lexicographically:
```python
max([("A", "B"), ("B", "A")])
```

### Unicode
In practice, we'll use bytes, not chars! (handle all languages, no out-of-vocabulary issues, etc.)
    - Vocab initialized with the 256 byte values and our special tokens (e.g. document boundaries)

Use `.encode('utf-8')` and `.decode('utf-8')` to move between bytes and strings
```python
# Encoding strings and ints
>>> "camfer".encode('utf-8')
>>> tuple("camfer".encode('utf-8')) # get byte values for encoding string
>>> "Hello, 🌍! 你好!".encode('utf-8')
>>> {i: bytes([i]) for i in range(10)}

# Decoding back
>>> secret = [99, 97, 109, 102, 101, 114, 33]
>>> "".join(bytes([b]).decode("utf-8") for b in secret)
```

### Pre-tokenization
If we run BPE naively on raw text, we might end up with things like:
```python
assert tokenize('dog') == 932
assert tokenize('dog!') == 1523
```

Ideally, we would preserve 'dog' and just have '!' as a token! This leads to pre-tokenization:
```python
import regex
GPT2 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
regex.findall(GPT2, "pre-tokenize this camfy!")
```
Tip: try not to use `.findall`, something like `.finditer` is preferred (why?)