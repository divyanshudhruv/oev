from oev.tokenizer import VOCAB_SIZE, PAD_ID, CLS_ID, SEP_ID, ANCHOR_ID, encode, decode


def test_vocab_size():
    assert VOCAB_SIZE == 99


def test_special_ids():
    assert (PAD_ID, CLS_ID, SEP_ID, ANCHOR_ID) == (0, 1, 2, 3)


def test_roundtrip():
    s = "Payment failed twice! [urgent] 4/5"
    assert decode(encode(s)) == s


def test_unknown_chars_dropped():
    assert decode(encode("café\n")) == "caf"
