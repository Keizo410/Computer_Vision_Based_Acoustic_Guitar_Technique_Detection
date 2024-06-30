from torchsummary import summary
from pyimagesearch.lenet import MultiStreamCNN

def custom_summary(model, input_size):
  """
  Prints a summary of the MultiStreamCNN model architecture.

  Args:
      model: The MultiStreamCNN model instance.
      input_size: A tuple representing the input size (e.g., (num_channels, 32, 32)).
  """

  # Extract LeNet details (assuming all LeNet instances have same architecture)
  lenet = model.hist_cnns[0]  # Access any LeNet instance from the module list

  print("---------- LeNet Summary ----------")
  print(f"Input Size: {input_size}")

  # Count parameters for each LeNet layer
  total_params_lenet = 0
  for name, param in lenet.named_parameters():
    if 'bias' in name:
      params = param.numel()
    else:
      params = param.numel() * param.size(1)  # Consider filter size for conv layers
    total_params_lenet += params
    print(f"{name}: {params} parameters")

  print(f"Total Parameters (LeNet): {total_params_lenet}")

  # Overall MultiStreamCNN details
  print("\n---------- MultiStreamCNN Summary ----------")
  total_params_multistream = total_params_lenet * len(model.hist_cnns)  # Total LeNet params * number of streams
  total_params_multistream += model.fc.in_features * model.fc.out_features  # FC layer params
  print(f"Total Trainable Parameters: {total_params_multistream}")

  # Print layer names (excluding weights and biases details)
  print("\n---------- Model Layers ----------")
  for name, _ in model.named_children():
    print(name)

  print("-" * 80)  # Separator

# Example usage
model = MultiStreamCNN(numChannels=1, num_classes=3)
custom_summary(model, input_size=(1, 32, 32))