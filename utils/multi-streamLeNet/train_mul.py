import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model.lenet import MultiStreamCNN
from torch import nn, optim
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, Grayscale
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

class CustomDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
        self.transform = transform

    def _load_samples(self):
        samples = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for case_dir in os.listdir(class_dir):
                case_path = os.path.join(class_dir, case_dir)
                histograms = self._load_histograms(case_path)
                samples.append((histograms, self.class_to_idx[cls]))
        return samples

    def _load_histograms(self, case_path):
        histograms = []
        for i in range(9):
            hist_path = os.path.join(case_path, f"{i}.png")
            histogram = Image.open(hist_path)
            histograms.append(histogram)
        return histograms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        histograms, label = self.samples[idx]
        if self.transform:
            histograms = [self.transform(hist) for hist in histograms]
        return histograms, label

class CustomDatasetForPrediction(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = self._load_samples()
        self.transform = transform

    def _load_samples(self):
        samples = []
        case_path = self.root_dir
        histograms = self._load_histograms(case_path)
        samples.append(histograms)
        return samples

    def _load_histograms(self, case_path):
        histograms = []
        for i in range(9):
            hist_path = os.path.join(case_path, f"{i}.png")
            histogram = Image.open(hist_path)
            histograms.append(histogram)
        return histograms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        histograms = self.samples[idx]
        if self.transform:
            histograms = [self.transform(hist) for hist in histograms]
        return histograms

def train_mult(dataset_root, output_root, classes):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 40
    num_classes = len(classes)

    # Define transformations
    transform = Compose([
        transforms.Resize((32, 32)),
        Grayscale(num_output_channels=1),  # Convert images to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create dataset
    dataset = CustomDataset(root_dir=dataset_root, classes=classes, transform=transform)
    train_data, val_data, train_labels, val_labels = train_test_split(dataset, dataset.samples, test_size=0.30, random_state=42)

    num_train_samples = len(train_labels)
    num_val_samples = len(val_labels)

    print(f'Number of training samples: {num_train_samples}')
    print(f'Number of validation samples: {num_val_samples}')

    # Create separate DataLoaders for training and validation
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)  # Shuffle not recommended for validation

    # Initialize model
    # model = MultiStreamCNN(numChannels=1 ,num_classes=3).to(device)
    model = MultiStreamCNN(numChannels=1, num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []  # List to store training loss for each epoch
    train_accuracy = []

    # Training loop
    for epoch in range(num_epochs):
        for i, (hist_streams, labels) in enumerate(train_loader):
            
            hist_streams = [stream.to(device) for stream in hist_streams]
            labels = labels.to(device)

            # Forward pass
            outputs = model(hist_streams)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss.append(loss.item())

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
             # Calculate and store training accuracy
            with torch.no_grad():
                correct = 0
                total = 0
                for hist_streams, labels in train_loader:
                    hist_streams = [stream.to(device) for stream in hist_streams]
                    labels = labels.to(device)
                    outputs = model(hist_streams)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_accuracy.append(correct / total)

    # Save the trained model
    model_path = os.path.join(output_root, "multi_stream_cnn.pth")
    torch.save(model.state_dict(), model_path)

    # Plotting (using matplotlib)
    import matplotlib.pyplot as plt

    # Plotting (using matplotlib)
    plt.subplot(1, 2, 1)  # Create subplot for loss curve
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')

    plt.subplot(1, 2, 2)  # Create subplot for accuracy curve
    plt.plot(train_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Curve')

    plt.tight_layout()  # Adjust spacing between subplots

    plt.savefig(f"{output_root}/train_mult_loss_acc_with_norm.png")  # Save combined plot

    
    evaluate(model_path, val_loader, num_classes)  # Pass the training data loader here (consider using a separate validation set)
   

# Function to evaluate the model
def evaluate(model_path, val_loader, num_classes):
    """
    This function evaluates the model performance on the data_loader.

    Args:
        model: The multistream CNN model.
        data_loader: The data loader for the evaluation set.
        device: The device ("cuda" or "cpu") to use for computation.

    Returns:
        A tuple containing accuracy and loss (optional, depending on your needs).
    """
    # Example usage after the training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Load the trained model from checkpoint
    state_dict = torch.load(model_path)

    # Create a new model instance (assuming you have the model definition)
    model = MultiStreamCNN(numChannels=1, num_classes=num_classes)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move the model to the chosen device (if necessary)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Create data loader
    data_loader = val_loader
    model.eval()  # Set the model to evaluation mode
    num_correct = 0
    num_samples = 0
    loss_val = 0.0
    all_predictions = []  # List to store all predicted labels
    all_ground_truths = []  # List to store all ground truth labels
    
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for i, (hist_streams, labels) in enumerate(data_loader):
            hist_streams = [stream.to(device) for stream in hist_streams]
            labels = labels.to(device)

            outputs = model(hist_streams)
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest probability

            num_correct += (predicted == labels).sum().item()
            num_samples += labels.size(0)
            loss_val += criterion(outputs, labels).item()

            # Store predictions and ground truths
            all_predictions.extend(predicted.cpu().numpy())  # Convert to numpy for easier handling
            all_ground_truths.extend(labels.cpu().numpy())
            
    accuracy = num_correct / num_samples

    # Calculate precision, recall, F1-score using scikit-learn
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_ground_truths, all_predictions, average=None)  # Weighted average for multi-class

    # Print or return metrics
    print(f'Evaluation Accuracy: {accuracy:.4f}')
    for i in range(len(precision)):
        print(f'Class {i}:')
        print(f'  Precision: {precision[i]:.4f}')
        print(f'  Recall: {recall[i]:.4f}')
        print(f'  F1-Score: {f1_score[i]:.4f}')
        print()

    print("##########################################################################\n")
    print()

    # Calculate precision, recall, F1-score using scikit-learn
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_ground_truths, all_predictions, average="weighted")  # Weighted average for multi-class

    print(f'Evaluation Precision (Weighted Average): {precision.mean():.4f}')
    print(f'Evaluation Recall (Weighted Average): {recall.mean():.4f}')
    print(f'Evaluation F1-Score (Weighted Average): {f1_score.mean():.4f}')

def predict(folder_path, num_classes):
    """
    This function predicts labels for data in the specified folder using a trained multi-stream CNN model.

    Args:
        model_path: The path to the saved model checkpoint.
        folder_path: The path to the folder containing data for prediction.

    Returns:
        A tuple containing accuracy, precision, recall, and F1-score.
    """
    model_path="./output/multi_stream_cnn.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model from checkpoint
    state_dict = torch.load(model_path)

    # Create a new model instance
    model = MultiStreamCNN(numChannels=1, num_classes=num_classes)  # Assuming you have the model definition

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move the model to the chosen device
    model.to(device)

    # Define transformations (modify as needed)
    transform = Compose([
        transforms.Resize((32, 32)),
        Grayscale(num_output_channels=1),  # Convert images to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load data for prediction
    dataset = CustomDatasetForPrediction(folder_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for i, hist_streams in enumerate(data_loader):
            hist_streams = [stream.to(device) for stream in hist_streams]

            outputs = model(hist_streams)
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest probability
    return predicted

def main():
    dataset_root = "../../../guitar_technique_detection/data/histgram_of_motion_dataset"
    output_root = "../../../guitar_technique_detection/model"
    classes = ["ham", "pull", "slide"]
    train_mult(dataset_root, output_root, classes)

if __name__ == "__main__":
    main()