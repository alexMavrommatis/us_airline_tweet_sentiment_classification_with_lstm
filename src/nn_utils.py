#==========================================================================
# A python library that provide utility code for visualization and
# and reporting results of an ANN classifier
#==========================================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
'''Visualization'''
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def plot_training_history(train_loss, test_loss):
    """"A function that plots the training and testing loss.

    Args:
    train_loss(List[float]): list storing the training loss values for each epoch.
    test_lossList[float]):  list storing the training MAE values for each epoch.

    """

    # 1) Plot Loss History
    plt.figure(figsize=(8,6))
    plt.plot(train_loss, label="Train Loss", color='orange')
    plt.plot(test_loss, label="Test Loss", color='blue')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()



# This function reports the names, shapes, and total count of learnable
# parameters in a PyTorch model.
def report_model_parameters(model):
    """"This function reports the names, shapes and total count
        learnable parameters in a Pytorch model.
    Args:
        model(nn.Module): The PyTorch neural network.

    """
    print("Model Parameters:")

    total_params = 0  # Counter for the total number of learnable parameters

    # Loop over all named parameters of the model
    for name, param in model.named_parameters():
        shape = tuple(param.shape)  # Get the shape as a tuple
        count = param.numel()       # Get the number of elements in the tensor
        total_params += count       # Accumulate total parameter count
        print(f"{name:40s} -> {shape}")

    print(f"\nTotal number of learnable parameters: {total_params}")

def report_validation_metrics(
    y_true,
    y_pred,
    target_names=['negative', 'neutral', 'positive'],
    case='Validation'
    ):
    """Computes and prints classification metrics for model evaluation.

    Args:
        y_true (array-like): Ground-truth class labels.
        y_pred (array-like): Predicted class labels from the model.
        labels (array-like): List of unique class labels (e.g., [0, 1, 2]).
        average (str): Averaging strategy for precision, recall, and F1.
        Options: 'micro', 'macro', 'weighted'. Default is 'macro'.
    """
    print(f"\n{case:-^60}")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{case} Confusion Matrix')
    plt.tight_layout()
    plt.show()