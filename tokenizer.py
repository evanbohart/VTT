def tokenize(text):
    tokens = []

    for word in text.split():
        token = "".join(
            char.lower() if char.isalnum() else ''
            for char in word
        )

        if token and token not in tokens:
            tokens.append(token)

    return tokens

def generate_tokens(df):
    vocab = {}

    vocab["<BOS>"] = 0
    vocab["<EOS>"] = 1

    next_id = 2

    for sentence in df['transcript']:
        for token in tokenize(str(sentence)):
            if token and token not in vocab:
                vocab[token] = next_id
                next_id += 1

    return vocab
