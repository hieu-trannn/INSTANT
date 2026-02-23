from evaluate import load

dataset_to_num_labels = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "mnli": 3
} #Number of classes in each dataset

dataset_fields = {
    "cola": ["sentence"],
    "sst2": ["sentence"],
    "mrpc": ["sentence1", "sentence2"],
    "qqp": ["question1", "question2"],
    "mnli": ["premise", "hypothesis"],
    "qnli": ["question", "sentence"],
    "rte": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}

dataset_best_metrics = {
    "cola": "matthews_correlation",  # Matthews correlation coefficient
    "sst2": "accuracy",  # Accuracy for binary classification
    "mrpc": "f1",  # F1 score for binary classification
    "qqp": "f1",  # F1 score for binary classification
    "mnli": "accuracy",  # Accuracy for multi-class classification
    "qnli": "accuracy",  # Accuracy for binary classification
    "rte": "accuracy",  # Accuracy for binary classification
    "wnli": "accuracy",  # Accuracy for binary classification
}

def compute_metrics_with_args(args):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1) if predictions.shape[-1] > 1 else predictions.squeeze()

        best_metric = dataset_best_metrics.get(args.dataset, "accuracy")

        metric = load(best_metric)

        return metric.compute(predictions=preds, references=labels)

    return compute_metrics

