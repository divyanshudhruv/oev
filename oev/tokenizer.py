SPECIALS = ["<pad>", "<cls>", "<sep>", "<anchor>"]
CHARS = [chr(i) for i in range(32, 127)]


def build_vocab():
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for c in CHARS:
        vocab[c] = len(vocab)
    return vocab


VOCAB = build_vocab()
VOCAB_SIZE = len(VOCAB)
PAD_ID = VOCAB["<pad>"]
CLS_ID = VOCAB["<cls>"]
SEP_ID = VOCAB["<sep>"]
ANCHOR_ID = VOCAB["<anchor>"]


def encode(text):
    return [VOCAB[c] for c in text if c in VOCAB]


def decode(ids):
    inv = {i: c for c, i in VOCAB.items()}
    return "".join(inv[i] for i in ids if i >= len(SPECIALS))
