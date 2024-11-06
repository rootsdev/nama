import json

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from src.data.filesystem import fopen
from transformers import PreTrainedTokenizerFast


def get_tokenize_function_and_vocab(
    use_phonemes=False,
    use_bigrams=False,
    use_edit_subwords=False,
    max_tokens=10,
    subwords_path=None,
    edit_subwords_path=None,
    subwords_bigrams_vocab_path=None,
    edit_subwords_bigrams_vocab_path=None,
    phoneme_vocab_path=None,
    phoneme_bigrams_vocab_path=None,
    nama_bucket=None,
):
    """ "Set up tokenizer and return tokenize function and vocab"""

    def tokenize_phonemes(name):
        return espeak.phonemize([name], separator=separator, strip=True)[0].split(" ")

    def tokenize_subwords(name):
        return subword_tokenizer.convert_ids_to_tokens(subword_tokenizer.encode(name))

    name_tokens_cache = {}

    if use_phonemes:
        with fopen(phoneme_bigrams_vocab_path if use_bigrams else phoneme_vocab_path, "r") as f:
            phoneme_vocab = json.load(f)
        espeak = EspeakBackend("en-us")
        separator = Separator(phone=" ", syllable=None, word="|")
        phoneme_vocab["[UNK]"] = len(phoneme_vocab)
        phoneme_vocab["[PAD]"] = len(phoneme_vocab)
        tokenizer = tokenize_phonemes
        tokenizer_vocab = phoneme_vocab
    else:
        if use_bigrams:
            path = edit_subwords_bigrams_vocab_path if use_edit_subwords else subwords_bigrams_vocab_path
            with fopen(path, "r") as f:
                subword_vocab = json.load(f)
            subword_tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
            tokenizer = tokenize_subwords
            tokenizer_vocab = subword_vocab
        else:
            path = edit_subwords_path if use_edit_subwords else subwords_path
            subword_tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
            tokenizer = tokenize_subwords
            tokenizer_vocab = subword_tokenizer.get_vocab()

    def tokenize(name):
        if name in name_tokens_cache:
            return name_tokens_cache[name]

        result = [tokenizer_vocab["[PAD]"]] * max_tokens
        unk = tokenizer_vocab["[UNK]"]
        tokens = tokenizer(name)
        context = "START"
        if use_bigrams:
            tokens.append("END")
        for ix, token in enumerate(tokens):
            if ix == max_tokens:
                break
            if use_bigrams:
                bigram = f"{context},{token}"
                result[ix] = tokenizer_vocab.get(bigram, tokenizer_vocab.get(token, unk))
            else:
                result[ix] = tokenizer_vocab.get(token, unk)
            context = token
        name_tokens_cache[name] = result

        return result

    return tokenize, tokenizer_vocab
