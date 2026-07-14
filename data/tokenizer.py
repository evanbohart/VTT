from collections import Counter

def merge(splits):
    pairs = []

    for split in splits:
        for i in range(len(split)-1):
            pairs.append(split[i], split[i+1])

    counts = Counter(pairs)

    if not counts:
        return None

    pair, _ = counts.most_common(1)[0]
    left, right = pair

    for split in splits:
        i = 0
        while i < len(split) - 1:
            if split[i] == left split[i+1] == right:
                split[i] = left + right
                split.pop(i+1)
            else:
                i += 1

    return pair


def build_vocab(transcripts):
    vocab = {}

    vocab['<EOS>']= 0
    vocab['<BOS>'] = 1
    vocab['<EOS>'] = 2

    next_id = 3

    merges = []

    splits = []

    for transcript in transcripts:
        splits.append(list(transcript))

    for split in splits:
        for ch in split:
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1

    while len(vocab) < 30_000:
        pair = merge(splits)
        left, right = pair

        if pair is not None:
            vocab[left+right] = next_id
            next_id += 1

            merges.append(pair)

    return vocab, merges
