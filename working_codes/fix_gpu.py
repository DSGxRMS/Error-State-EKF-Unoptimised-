import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
