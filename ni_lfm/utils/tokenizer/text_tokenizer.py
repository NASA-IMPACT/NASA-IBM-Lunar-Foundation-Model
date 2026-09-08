import json
import os

from collections import defaultdict

import numpy as np

from tokenizers import AddedToken, Tokenizer, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer


UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
S1_TOKEN = "[S_1]"


def generate_sentinel_tokens(num=100, start_id=0):
    tokens = [
        AddedToken(content=f"[S_{i}]", single_word=True, normalized=False)
        for i in range(start_id, num + start_id)
    ]

    return tokens


def generate_coord_tokens(bins=1000):
    tokens = []
    coords_str = ["xmin={}", "ymin={}", "xmax={}", "ymax={}"]

    for s in coords_str:
        for i in range(bins):
            tokens.append(AddedToken(content=s.format(i), single_word=True, normalized=False))

    return tokens


def generate_object_class_tokens(dataset="coco"):
    with open(os.path.join(os.path.dirname(__file__), "object_classes.json")) as f:
        object_classes = json.load(f)[dataset]

    tokens = [
        AddedToken(content=class_name, single_word=True, normalized=True)
        for class_name in object_classes
    ]

    return tokens


def train_unified_wordpiece_tokenizer(
    files,
    vocab_size,
    sentinel_tokens: list[str | AddedToken] | None = None,
    coord_tokens: list[str | AddedToken] | None = None,
    object_class_tokens: list[str | AddedToken] | None = None,
    unk_token: str | AddedToken = UNK_TOKEN,
    pad_token: str | AddedToken = PAD_TOKEN,
    sos_token: str | AddedToken = SOS_TOKEN,
    eos_token: str | AddedToken = EOS_TOKEN,
    additional_special_tokens: list[str | AddedToken] | None = None,
    min_frequency=0,
    clean_text: bool = True,
    handle_chinese_chars: bool = True,
    strip_accents: bool | None = None,
    lowercase: bool = True,
    wordpieces_prefix: str = "##",
    show_progress=True,
):
    tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token)))

    tokenizer.normalizer = BertNormalizer(
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

    special_tokens = []
    special_tokens.extend((pad_token, unk_token, sos_token, eos_token))

    if sentinel_tokens is not None:
        special_tokens.extend(sentinel_tokens)
    if coord_tokens is not None:
        special_tokens.extend(coord_tokens)
    if object_class_tokens is not None:
        special_tokens.extend(object_class_tokens)
    if additional_special_tokens is not None:
        special_tokens.extend(additional_special_tokens)

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        continuing_subword_prefix=wordpieces_prefix,
        special_tokens=special_tokens,
    )

    if isinstance(files, str):
        files = [files]

    tokenizer.train(files, trainer=trainer)

    return tokenizer


def encode_sequence(
    sequence: str | list[str],
    tokenizer: Tokenizer,
    max_tokens: int,
    eos_token: str = EOS_TOKEN,
) -> list[int] | list[list[int]]:
    """Encode a text sequence or sequence chunks using the text tokenizer."""
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"Tokenizer does not contain EOS token {eos_token}")

    if isinstance(sequence, str):
        # Tokenize the sequence and get the ids
        seq_ids = tokenizer.encode(sequence).ids
        # Add EOS to all sequences
        seq_ids.append(eos_id)
        # Truncate sequence
        seq_ids = seq_ids[:max_tokens]

    elif isinstance(sequence, list):
        # Tokenize the sequence chunks and get the ids
        encoded_seq_chunks = tokenizer.encode_batch(sequence)
        seq_chunks = [seq.ids for seq in encoded_seq_chunks]
        # Add EOS as an extra chunk
        seq_chunks.append([eos_id])
        # Truncate sequence to keep all chunks below max token length
        cumulative_token_count = np.cumsum(np.array([len(chunk) for chunk in seq_chunks]))
        seq_ids = [
            chunk
            for (chunk, token_count) in zip(seq_chunks, cumulative_token_count)
            if token_count <= max_tokens
        ]
    else:
        raise TypeError(f"Invalid sequence type: {sequence}")

    return seq_ids


def decode_token_sequences(
    token_ids,
    tokenizer: Tokenizer,
    skip_special_tokens: bool = True,
    pad_token: str = PAD_TOKEN,
) -> list[str]:
    """Decode token IDs to text strings.

    Args:
        token_ids: (B, L) or (B, L, num_codebooks) tensor of token IDs
        tokenizer: HuggingFace tokenizer
        skip_special_tokens: Whether to skip [PAD], [SOS], [EOS]
        pad_token: token used for padding

    Returns:
        List of decoded strings (one per batch item)
    """
    if hasattr(token_ids, "ndim") and token_ids.ndim == 3:
        token_ids = token_ids[:, :, 0]

    pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError(f"Tokenizer does not contain PAD token {pad_token}")

    decoded = []
    for t in range(len(token_ids)):
        ids = token_ids[t].cpu().tolist()
        ids = [token_id for token_id in ids if token_id != pad_id]
        sentence = []
        for i in ids:
            sentence.append(tokenizer.decode([i], skip_special_tokens=skip_special_tokens))
        text = " ".join([x for x in sentence if x != ""])
        decoded.append(text)
    return decoded


def get_sentinel_to_id_mapping(tokenizer, match_str="[S_"):
    sentinel_tokens = {k: v for k, v in tokenizer.get_vocab().items() if k.startswith(match_str)}
    # Extract the sentinel token id, the id is of the form "[S_0]", "[S_1]", etc.
    sentinel_to_id = {
        int(k.split("_")[1][:-1]): v
        for k, v in sorted(sentinel_tokens.items(), key=lambda x: x[1])
    }
    return sentinel_to_id


def split_by_sentinel(seq_ids, sentinel_ids):
    splits = defaultdict(list)
    cur_sentinel = None
    for token in seq_ids:
        if token in sentinel_ids:
            cur_sentinel = token
        else:
            splits[cur_sentinel].append(token)

    return splits


def merge_span_masking(input_seq, decoder_seq, sentinel_ids):
    decoder_splits = split_by_sentinel(decoder_seq, sentinel_ids)
    out_seq = []
    for token in input_seq:
        if token in sentinel_ids:
            out_seq.extend(decoder_splits[token])
        else:
            out_seq.append(token)
    return out_seq
