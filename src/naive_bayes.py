import math
from collections import Counter

from src.tokenizer import tokenize
from src.features import build_vocabulary
from src.features import create_bow_matrix

class NaiveBayes:

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, emails, labels):

        self.vocab, self.word_to_index = \
            build_vocabulary(emails)

        bow = create_bow_matrix(
            emails,
            self.word_to_index
        )

        V = len(self.vocab)

        total = len(labels)

        spam_count = labels.count("spam")
        ham_count = labels.count("ham")

        self.p_spam = spam_count / total
        self.p_ham = ham_count / total

        spam_wc = [0] * V
        ham_wc = [0] * V

        N_spam = 0
        N_ham = 0

        for row, label in zip(bow, labels):

            if label == "spam":

                for i in range(V):
                    spam_wc[i] += row[i]
                    N_spam += row[i]

            else:

                for i in range(V):
                    ham_wc[i] += row[i]
                    N_ham += row[i]

        self.p_w_spam = [

            (spam_wc[i] + self.alpha)
            /

            (N_spam + self.alpha * V)

            for i in range(V)
        ]

        self.p_w_ham = [

            (ham_wc[i] + self.alpha)
            /

            (N_ham + self.alpha * V)

            for i in range(V)
        ]

    def predict_one(self, email):

        counts = Counter(
            tokenize(email)
        )

        log_spam = math.log(
            self.p_spam
        )

        log_ham = math.log(
            self.p_ham
        )

        for word, count in counts.items():

            if word not in self.word_to_index:
                continue

            i = self.word_to_index[word]

            log_spam += (
                count
                *
                math.log(
                    self.p_w_spam[i]
                )
            )

            log_ham += (
                count
                *
                math.log(
                    self.p_w_ham[i]
                )
            )

        if log_spam > log_ham:
            return "spam"

        return "ham"

    def predict(self, emails):

        return [
            self.predict_one(email)
            for email in emails
        ]