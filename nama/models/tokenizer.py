from transformers import PreTrainedTokenizerFast


def get_tokenize_function_and_vocab(tokenizer_path, max_tokens):
    """ "Set up tokenizer and return tokenize function and vocab"""

    name_tokens_cache = {}

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer_vocab = tokenizer.get_vocab()

    def tokenize(name):
        if name in name_tokens_cache:
            return name_tokens_cache[name]

        result = [tokenizer_vocab["[PAD]"]] * max_tokens
        unk = tokenizer_vocab["[UNK]"]
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(name))
        for ix, token in enumerate(tokens):
            if ix == max_tokens:
                break
            result[ix] = tokenizer_vocab.get(token, unk)
        name_tokens_cache[name] = result

        return result

    return tokenize, tokenizer_vocab
