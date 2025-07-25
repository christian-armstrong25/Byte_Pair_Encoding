import os
import pathlib
from collections import Counter, defaultdict

import regex

from utils import GPT2_TOKENIZER_REGEX as PAT
from utils import get_chunks


def count_pre_tokens(chunk: str, r_split: str) -> dict | Counter:
    """
    Count occurrences of pre-tokens formed by PAT split (splitting into subwords before encoding to byte sequences)

    Returns:
        pre_token_counts: Counter[tuple[bytes], int]

    Example:
        - chunk: "hello world<|endoftext|>goodbye world"
        - r_split: "<|endoftext|>"
        - PAT: " "
        - sub_chunks = ["oh hi", "oh no"]
        - Pre-tokenization -> [["oh", "hi"], ["oh", "no"]]
        - []
        - [('o', 'h'), ]
        - [(b'o', b'h'), (b'h', b'i'), (b'n', b'o')]
        - pre_token_counts: {(b'o', b'h'): 2, (b'h', b'i'): 1, (b'n', b'o'): 1}
    """
    # Before pre-tokenization, we need to split on special tokens so these are not split/merged later
    sub_chunks = regex.split(r_split, chunk)
    pre_tokenization = [regex.findall(PAT, x) for x in sub_chunks]

    pre_tokenization = [tuple(bytes([byte]) for byte in x.encode('utf-8'))
                        for tokens in pre_tokenization for x in tokens]
    pre_token_counts = Counter()  # or recommended Counter()
    pre_token_counts = Counter(pre_tokenization)

    return pre_token_counts


def train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, train a BPE tokenizer and output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = os.cpu_count() or 1
    vocab = {i: bytes([i]) for i in range(256)}
    vocab = vocab | {i + 256: token.encode("utf-8")
                     for i, token in enumerate(special_tokens)}

    with open(input_path, "rb") as f:
        chunks = get_chunks(f, num_processes, "<|endoftext|>".encode("utf-8"))

    r_split = "|".join(
        # regex looks like |<\|token1\|>|<\|token2\|>|...
        regex.escape(token) for token in special_tokens)

    # TODO: YOUR CODE HERE
    # Get all the pre-token counts (using your function above and these chunks)
    pre_token_counts = [count_pre_tokens(chunk, r_split) for chunk in chunks]

    # TODO: YOUR CODE HERE
    # Get and sum the counts of all consecutive byte-pairs of tokens in each of the pre-tokens found above
    consecutive_pair_counts = defaultdict(int)
    for pre_token_counter in pre_token_counts:
        for token_tuple, count in pre_token_counter.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i+1])
                consecutive_pair_counts[pair] += count

    merges = []

    # TODO: YOUR CODE HERE
    # Implement the BPE algorithm here using your previous calculations
    while len(vocab) < vocab_size and consecutive_pair_counts:
        max_count = max(consecutive_pair_counts.values())
        # Get all pairs with the max count
        max_pairs = [
            pair for pair, count in consecutive_pair_counts.items() if count == max_count]
        # Break ties lexicographically
        max_pair = max(max_pairs)
        new_token = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        vocab[len(vocab)] = new_token

        for word in pre_token_counts:
            for token_tuple, count in list(word.items()):
                i = 0
                while i < len(token_tuple) - 1:
                    if (token_tuple[i], token_tuple[i+1]) == max_pair:
                        # Decrement the merged pair
                        consecutive_pair_counts[max_pair] -= count
                        if consecutive_pair_counts[max_pair] == 0:
                            del consecutive_pair_counts[max_pair]

                        # Decrement the pair before, if it exists
                        if i > 0:
                            prev_pair = (token_tuple[i-1], token_tuple[i])
                            consecutive_pair_counts[prev_pair] -= count
                            if consecutive_pair_counts[prev_pair] == 0:
                                del consecutive_pair_counts[prev_pair]

                        # Decrement the pair after, if it exists
                        if i + 2 < len(token_tuple):
                            next_pair = (token_tuple[i+1], token_tuple[i+2])
                            consecutive_pair_counts[next_pair] -= count
                            if consecutive_pair_counts[next_pair] == 0:
                                del consecutive_pair_counts[next_pair]

                        # Increment the new pairs
                        if i > 0:
                            new_prev_pair = (token_tuple[i-1], new_token)
                            consecutive_pair_counts[new_prev_pair] += count
                        if i + 2 < len(token_tuple):
                            new_next_pair = (new_token, token_tuple[i+2])
                            consecutive_pair_counts[new_next_pair] += count

                        # Update the token tuple
                        token_list = list(token_tuple)
                        token_list[i:i+2] = [new_token]
                        new_tuple = tuple(token_list)
                        word[new_tuple] = count
                        del word[token_tuple]
                        token_tuple = new_tuple
                        # After merging, do not increment i (since the tuple is now shorter)
                        if i > 0:
                            i -= 1
                    else:
                        i += 1

    print(vocab)

    return vocab, merges


if __name__ == "__main__":
    import json

    from tests import gpt2_bytes_to_unicode
    input_path = pathlib.Path(
        __file__).resolve().parent / "data" / "corpus.txt"
    vocab, merges = train(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Load reference merges
    reference_merges_path = pathlib.Path(__file__).resolve(
    ).parent / "data" / "train-bpe-reference-merges.txt"
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]

    # Compare merges
    for i, (m, r) in enumerate(zip(merges, reference_merges)):
        if m != r:
            print(f"First merge difference at index {i}:")
            print(f"  Your merge:      {m}")
            print(f"  Reference merge: {r}")
            break
    else:
        if len(merges) != len(reference_merges):
            print(
                f"Different number of merges: yours={len(merges)}, reference={len(reference_merges)}")
        else:
            print("All merges match.")

    # Load reference vocab
    reference_vocab_path = pathlib.Path(__file__).resolve(
    ).parent / "data" / "train-bpe-reference-vocab.json"
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token]
                                    for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    your_vocab_set = set(vocab.values())
    ref_vocab_set = set(reference_vocab.values())
    if your_vocab_set != ref_vocab_set:
        print("Vocab differences:")
        print("In your vocab but not in reference:",
              your_vocab_set - ref_vocab_set)
        print("In reference but not in your vocab:",
              ref_vocab_set - your_vocab_set)
    else:
        print("Vocab sets match.")

    print("Done!")
