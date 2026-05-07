def accuracy_score(
    predictions,
    labels
):

    correct = sum(

        pred == true

        for pred, true
        in zip(
            predictions,
            labels
        )
    )

    return correct / len(labels)


def confusion_matrix(
    predictions,
    labels
):

    tp = sum(

        p == "spam"
        and
        t == "spam"

        for p, t
        in zip(
            predictions,
            labels
        )
    )

    tn = sum(

        p == "ham"
        and
        t == "ham"

        for p, t
        in zip(
            predictions,
            labels
        )
    )

    fp = sum(

        p == "spam"
        and
        t == "ham"

        for p, t
        in zip(
            predictions,
            labels
        )
    )

    fn = sum(

        p == "ham"
        and
        t == "spam"

        for p, t
        in zip(
            predictions,
            labels
        )
    )

    return tp, tn, fp, fn