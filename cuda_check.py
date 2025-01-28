import torch
print('GPU available?',torch.cuda.is_available())

# Check if PyTorch can access GPUs
gpus = torch.cuda.device_count()

if gpus > 0:
    print(f"Number of GPUs available: {gpus}")
else:
    print("No GPUs available.")