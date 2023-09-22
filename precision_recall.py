def compute_precision_recall(resultTestData, labeledTestData):
    
    tf_result = []
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for label, prediction in zip(resultTestData['predicted_label'], labeledTestData['label']):
        if label == 'ham' and prediction == 'ham':
            TP += 1
            tf_result.append('TP')
        elif label == 'ham' and prediction == 'spam':
            FN += 1
            tf_result.append('TN')
        elif label == 'spam' and prediction == 'ham':
            FP += 1
            tf_result.append('FP')
        elif label == 'spam' and prediction == 'spam':
            TN += 1
            tf_result.append('FN')
    
    precision = compute_precision(TP, FP);
    recall = compute_recall(TP, FN);
    
    print("Number of TP:", TP)
    print("Number of TN:", TN)
    print("Number of FP:", FP)
    print("Number of FN:", FN)
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2)) 

    # Save the predicted data to a CSV file
    result_file_path = "Results/AtolePrecisionRecall.csv"
    resultTestData.insert(0, 'measure', tf_result)
    resultTestData.insert(1, 'correet_label', labeledTestData['label'])
    resultTestData.to_csv(result_file_path, index=False)
    print(f"Predicted data saved to {result_file_path}")
    
def compute_precision(TP, FP):
    print("Computing precision...")
    return TP / (TP + FP)

def compute_recall(TP, FN):
    print("Computing recall...")
    return TP / (TP + FN)