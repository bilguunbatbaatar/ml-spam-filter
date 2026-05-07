from tokenizer import tokenize


def build_vocabulary(emails):

    vocab_set = set()

    for email in emails:
        vocab_set.update(tokenize(email))

    vocab = sorted(list(vocab_set))

    word_to_index = {
        word: i
        for i, word in enumerate(vocab)
    }

    return vocab, word_to_index


def create_bow_matrix(emails, word_to_index):

    matrix = []

    for email in emails:

        row = [0] * len(word_to_index)

        for word in tokenize(email):

            if word in word_to_index:
                row[word_to_index[word]] += 1

        matrix.append(row)

    return matrix