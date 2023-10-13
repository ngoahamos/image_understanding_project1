import numpy as np;
def evaluate_performance_metric(ground_truth_points, detected_points):
    # print(detected_points)
    # return 0,0,0
    true_positives = np.sum(np.in1d(detected_points, ground_truth_points))
    print("true positives", true_positives)
    false_positives = len(detected_points) - true_positives
    print("false positives", false_positives)
    false_negative = len(ground_truth_points) - true_positives
    print("false negatives", false_negative)

    precision = (true_positives) / (true_positives + false_positives)
    recall = (true_positives) / (true_positives + false_negative)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
