import torch 
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn, optim
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, Grayscale
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
import yaml 
import os 

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    state_dict = torch.load(model_path)

    model = MultiStreamCNN(numChannels=1, num_classes=num_classes)

    model.load_state_dict(state_dict)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    data_loader = val_loader
    model.eval()  
    num_correct = 0
    num_samples = 0
    loss_val = 0.0
    all_predictions = []  
    all_ground_truths = []  
    
    with torch.no_grad():  
        for i, (hist_streams, labels) in enumerate(data_loader):
            hist_streams = [stream.to(device) for stream in hist_streams]
            labels = labels.to(device)

            outputs = model(hist_streams)
            _, predicted = torch.max(outputs.data, 1)  
            num_correct += (predicted == labels).sum().item()
            num_samples += labels.size(0)
            loss_val += criterion(outputs, labels).item()

            all_predictions.extend(predicted.cpu().numpy())  
            all_ground_truths.extend(labels.cpu().numpy())
            
    accuracy = num_correct / num_samples

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_ground_truths, all_predictions, average=None)  

    print(f'Evaluation Accuracy: {accuracy:.4f}')
    for i in range(len(precision)):
        print(f'Class {i}:')
        print(f'  Precision: {precision[i]:.4f}')
        print(f'  Recall: {recall[i]:.4f}')
        print(f'  F1-Score: {f1_score[i]:.4f}')
        print()

    print("##########################################################################\n")
    print()

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_ground_truths, all_predictions, average="weighted")  

    print(f'Evaluation Precision (Weighted Average): {precision:.4f}')
    print(f'Evaluation Recall (Weighted Average): {recall:.4f}')
    print(f'Evaluation F1-Score (Weighted Average): {f1_score:.4f}')

    
def train(dataset_root, output_root, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 40
    num_classes = len(classes)

    transform = Compose([
        transforms.Resize((32,32)),
        Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = CustomDataset(root_dir = dataset_root, classes = classes, transform = transform)
    train_data, val_data, train_labels, val_labels = train_test_split(dataset, dataset.samples, test_size=0.30, random_state=42)
    num_train_samples = len(train_labels)
    num_val_samples = len(val_labels)

    print(f'Number of training samples: {num_train_samples}')
    print(f'Number of validation samples: {num_val_samples}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)  

    model = MultiStreamCNN(numChannels=1, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []  
    train_accuracy = []

    for epoch in range(num_epochs):
        for i, (hist_streams, labels) in enumerate(train_loader):
            
            hist_streams = [stream.to(device) for stream in hist_streams]
            labels = labels.to(device)

            outputs = model(hist_streams)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
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

    model_path = os.path.join(output_root, "multi_stream_cnn.pth")
    torch.save(model.state_dict(), model_path)

    plt.subplot(1, 2, 1)  
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')

    plt.subplot(1, 2, 2)  
    plt.plot(train_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Curve')

    plt.tight_layout()  

    plt.savefig(f"{output_root}/train_mult_loss_acc_with_norm.png")  

    evaluate(model_path, val_loader, num_classes)


def main():
    BASE_DIR = Path.cwd().parent.parent
    
    with open(BASE_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    dataset_root = BASE_DIR / config["histogram_of_motion_output_data_path"]
    output_root = BASE_DIR / config["model_weight_output_path"]
    classes = ["ham", "pull", "slide"]
    train(dataset_root, output_root, classes)

if __name__ == "__main__":
    main()