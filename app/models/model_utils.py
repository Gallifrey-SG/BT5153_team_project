''' 
provides utilities to predict outcomes using a model
compute and display various evaluation metrics
determine optimal thresholds based on precision-recall criteria
find the optimal threshold 
plotting the precision-recall curve, roc curve,and confusion matrix to evaluate model performance visually

the model class can handle additional metadata features if provided and operates on specified computing devices
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (precision_recall_curve, roc_curve, auc, roc_auc_score,
                             precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)

class ModelUtility:
    def __init__(self, model, device, acceptable_recall=0.70, precision_weight=0.3):
        '''
        initializes the model utility class with a model, computing device, and metrics for evaluation thresholds.

        :param model: the machine learning model to be used for predictions.
        :param device: the device (cpu or gpu) on which the model computations will be performed.
        :param acceptable_recall: the minimum recall value considered acceptable for determining the optimal threshold.
        :param precision_weight: the weight given to precision in the calculation of the f-score, influencing threshold choice.
        '''
        self.model = model
        self.device = device
        self.acceptable_recall = acceptable_recall
        self.precision_weight = precision_weight

    def model_predict(self, test_loader, return_labels=False):
        '''
        performs predictions on a given dataset using the model, and optionally returns true labels along with predicted probabilities and labels.

        :param test_loader: the data loader containing the test dataset.
        :param return_labels: flag indicating whether to return the true labels along with predictions.
        :return: a tuple containing predicted probabilities, predicted labels, and optionally true labels.
        '''
        self.model.eval()  # Set the model to evaluation mode
        true_labels = []
        predicted_probs = []
        predicted_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
                }

                if 'meta_features' in batch:
                    inputs['meta_features'] = batch['meta_features'].to(self.device)

                if return_labels:
                    labels = batch['labels'].to(self.device)

                outputs = self.model(**inputs)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

                probs = torch.softmax(logits, dim=1)
                pos_probs = probs[:, 1]
                preds = torch.argmax(probs, dim=1)

                predicted_probs.extend(pos_probs.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())
                if return_labels:
                    true_labels.extend(labels.cpu().numpy())

            if return_labels:
                return predicted_probs, predicted_labels, true_labels
            else:
                return predicted_probs, predicted_labels, None
            
    def threshold_precision(self, true_labels, probabilities):
        '''
        calculates the optimal threshold for predictions based on the precision that meets the acceptable recall.

        :param true_labels: the actual labels of the test set.
        :param probabilities: the predicted probabilities for the positive class.
        :return: the optimal threshold value if an acceptable recall level is found, otherwise None.
        '''
        precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
        auprc = auc(recall, precision)
        print(f"AUPRC: {auprc:.4f}")

        auroc = roc_auc_score(true_labels, probabilities)
        print(f"AUROC: {auroc:.4f}")

        acceptable_indices = np.where(recall >= self.acceptable_recall)[0]
        if len(acceptable_indices) > 0:
            optimal_idx = acceptable_indices[np.argmax(precision[acceptable_indices])]
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal Threshold: {optimal_threshold:.4f}")
            return optimal_threshold
        else:
            print("No threshold found with the recall above or equal to the acceptable level.")
            return None

    def threshold_recall(self, true_labels, probabilities):
        '''
        calculates the optimal threshold for predictions based on the best f-score which considers both precision and recall,
        using the specified precision weight to adjust the importance given to precision.

        :param true_labels: the actual labels of the test set.
        :param probabilities: the predicted probabilities for the positive class.
        :return: the optimal threshold value if an acceptable recall level is found, otherwise None.
        '''
        precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
        auprc = auc(recall, precision)
        print(f"AUPRC: {auprc:.4f}")

        auroc = roc_auc_score(true_labels, probabilities)
        print(f"AUROC: {auroc:.4f}")

        # Calculate the F-score with the given precision weight
        if self.precision_weight < 1:
            beta = np.sqrt((1 - self.precision_weight) / self.precision_weight)
            fscore = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        else:
            fscore = 2 * (precision * recall) / (precision + recall)  # When weight is 1, it's the regular F1 score

        fscore[np.isnan(fscore)] = 0  # Handle the case where precision and recall are both zero

        acceptable_indices = np.where(recall >= self.acceptable_recall)[0]
        if len(acceptable_indices) > 0:
            optimal_idx = acceptable_indices[np.argmax(fscore[acceptable_indices])]
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal Threshold: {optimal_threshold:.4f} with F-score: {fscore[optimal_idx]:.4f}")
            return optimal_threshold
        else:
            print("No threshold found with the recall above or equal to the acceptable level.")
            return None

    def model_test(self, true_labels, probabilities, threshold=0.50):
        '''
        evaluates the model performance using specified threshold, calculates various metrics and displays plots for analysis.

        :param true_labels: the actual labels of the test dataset.
        :param probabilities: the predicted probabilities for the positive class.
        :param threshold: the threshold to be used for converting probabilities to binary predictions.
        :return: displays performance metrics and visual plots including precision-recall curve, roc curve, and confusion matrix.
        '''
        probabilities = np.array(probabilities)

        precision, recall, thresholds_prc = precision_recall_curve(true_labels, probabilities)
        auprc = auc(recall, precision)
        print(f"AUPRC: {auprc:.4f}")

        fpr, tpr, thresholds_roc = roc_curve(true_labels, probabilities)
        auroc = roc_auc_score(true_labels, probabilities)
        print(f"AUROC: {auroc:.4f}")

        predictions = (probabilities >= threshold).astype(int)

        precision_score_val = precision_score(true_labels, predictions)
        recall_score_val = recall_score(true_labels, predictions)
        f1_score_val = f1_score(true_labels, predictions)
        accuracy_score_val = accuracy_score(true_labels, predictions)

        print(f"Threshold: {threshold:.4f}")
        print(f"Precision: {precision_score_val:.4f}")
        print(f"Recall: {recall_score_val:.4f}")
        print(f"F1 Score: {f1_score_val:.4f}")
        print(f"Accuracy: {accuracy_score_val:.4f}")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Precision-Recall Curve
        ax1.plot(recall, precision, label=f'PR Curve (area = {auprc:.2f})')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend(loc="best")
        ax1.grid(True)

        # ROC Curve
        ax2.plot(fpr, tpr, label=f'ROC Curve (area = {auroc:.2f})')
        ax2.plot([0, 1], [0, 1], 'r--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="best")
        ax2.grid(True)

        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues, ax=ax3)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted Labels')
        ax3.set_ylabel('True Labels')

        plt.tight_layout()
        plt.show()
