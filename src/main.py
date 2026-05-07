from src.dataset import read_dataset
from src.naive_bayes import NaiveBayes
from src.metrics import accuracy_score
from src.metrics import confusion_matrix

TRAIN_PATH = "data/train"
DEV_PATH = "data/dev"


train_emails, train_labels = \
    read_dataset(TRAIN_PATH)

dev_emails, dev_labels = \
    read_dataset(DEV_PATH)


model = NaiveBayes(
    alpha=1
)

model.fit(
    train_emails,
    train_labels
)


predictions = model.predict(
    dev_emails
)


accuracy = accuracy_score(
    predictions,
    dev_labels
)


tp, tn, fp, fn = \
    confusion_matrix(
        predictions,
        dev_labels
    )


print(
    f"Accuracy: {accuracy:.2%}"
)

print(
    f"TP={tp} "
    f"TN={tn} "
    f"FP={fp} "
    f"FN={fn}"
)