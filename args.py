import torch

learning_rate = 1e-3
batch_size = 32
num_epochs = 250
momentum = 0.9
num_workers = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_decay = 1e-5
image_dimension = 28
save_path = "./saves"
data_path = "./data"
