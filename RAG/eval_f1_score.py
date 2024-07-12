from collections import Counter
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(predictions, ground_truths):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for prediction, ground_truth in zip(predictions, ground_truths):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common > 0:
            true_positives = 1
            false_positives = 0
            false_negatives = 0
        else:
            true_positives = 0
            false_positives = 1
            false_negatives = 1
        
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    precision = 1.0 * total_true_positives / (total_true_positives + total_false_positives)
    recall = 1.0 * total_true_positives / (total_true_positives + total_false_negatives)
    f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
    
    return f1





