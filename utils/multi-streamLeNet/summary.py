from torchsummary import summary

def custom_summary(model, input_size):
  """
  Prints a summary of the MultiStreamCNN model architecture.

  Args:
      model: The MultiStreamCNN model instance.
      input_size: A tuple representing the input size (e.g., (num_channels, 32, 32)).
  """

  lenet = model.streams[0]  

  print("---------- LeNet Summary ----------")
  print(f"Input Size: {input_size}")

  total_params_lenet = 0
  for name, param in lenet.named_parameters():
    if 'bias' in name:
      params = param.numel()
    else:
      params = param.numel() * param.size(1)  
    total_params_lenet += params
    print(f"{name}: {params} parameters")

  print(f"Total Parameters (LeNet): {total_params_lenet}")

  print("\n---------- MultiStreamCNN Summary ----------")
  total_params_multistream = total_params_lenet * len(model.streams)  
  total_params_multistream += model.classifier.in_features * model.classifier.out_features  
  print(f"Total Trainable Parameters: {total_params_multistream}")

  print("\n---------- Model Layers ----------")
  for name, _ in model.named_children():
    print(name)

  print("-" * 80)  

model = MultiStreamCNN(numChannels=1, num_classes=3)
custom_summary(model, input_size=(1, 32, 32))