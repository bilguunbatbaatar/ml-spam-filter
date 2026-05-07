import os


def read_dataset(folder_path):
    emails = []
    labels = []

    for label in ["spam", "ham"]:
        label_folder = os.path.join(folder_path, label)

        for filename in os.listdir(label_folder):

            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(label_folder, filename)

            with open(
                filepath,
                "r",
                encoding="latin-1",
                errors="ignore"
            ) as f:

                emails.append(f.read())
                labels.append(label)

    return emails, labels