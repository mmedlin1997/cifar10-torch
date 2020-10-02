import torch
import torchvision
from torchvision import datasets, transforms  # image data, transform to torch tensor format 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
from datetime import timedelta
import argparse

from platform import python_version
print("python", python_version())
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("matplotlib", matplotlib.__version__)
print("seaborn", sns.__version__)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d","--device", help="device type(ex. cpu, cuda, cuda:0)")
args = parser.parse_args()

if args.device == "cpu" or args.device == "cuda" or args.device == "cuda:0":
  print("Requested device:", args.device)
else:
  print("Using default device")

# Set CPU or GPU if availabe
if torch.cuda.is_available() == True:
  count = torch.cuda.device_count()
  print("GPU number of devices:", count)
  print(*["GPU device["+str(x)+"]="+torch.cuda.get_device_name(x) for x in range(count)], sep="\n")
  print("GPU current device:", torch.cuda.current_device())

# device can be int or string0
device = torch.device('cuda:0' if torch.cuda.is_available() and not (args.device == 'cpu') else 'cpu')
print('Using device:', device)

# Download datasets
test = datasets.MNIST('', train=False, download=True, 
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

# Create a class to define the NN model.
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    # Define layers
    self.fc1 = nn.Linear(28*28, 64) # Linear - fully-connected layer (input, output). This layer is input, designed to take a single image.
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 10)

  # Define how data flows forward through nn, and activations
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    logits = self.fc4(x)
    out = F.log_softmax(logits, dim=1)
    return out

model = Model().to(device)

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load model as PyTorch
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#print("Load model:", checkpoint.keys())
#print("Load model:", checkpoint['model_state_dict'].keys())
#print("Load model:", checkpoint['optimizer_state_dict'].keys())

# Results
# Inspect batch
# NOTE: images predicted on GPU if available, then moved to CPU for matplotlib plot 
elapsed_time = 0.0
batch_size = 300
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
plt.figure(figsize=(10,6))
for test_images, test_labels in test_loader:
  start = timer()
  output = model(test_images.to(device).view(-1,784)).cpu()
  end = timer()  
  elapsed_time += end-start
  for i, img in enumerate(output):
    expected = test_labels[i]
    inferred = torch.argmax(img)
    cmap = plt.cm.binary if expected == inferred else plt.cm.autumn
    plt.subplot(int(batch_size/20), 20, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].view(28,28), cmap=cmap) 
    plt.xlabel(expected.item())
    plt.suptitle('Batch of ' + str(batch_size), fontsize=16, y=.9)
  break
plt.show()
print("Total prediction time[", batch_size, "]:", timedelta(seconds=elapsed_time))
print("Avg time / prediction: ", timedelta(seconds=elapsed_time)/batch_size)

#%% Confusion Matrix
# Inspect batch
batch_size = 100
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

num_classes = 10
class_names = [i for i in range(10)]
cf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.uint8)

for test_images, test_labels in test_loader:
  output = model(test_images.to(device).view(-1,784)).cpu()  
  for i, img in enumerate(output):
    expected = test_labels[i]
    inferred = torch.argmax(img)
    cf_matrix[expected][inferred] += 1

plt.figure(figsize=(10,6))
ax = sns.heatmap(cf_matrix, annot=True, 
                 yticklabels=class_names, xticklabels=class_names, fmt='', 
                 linewidths=1, linecolor='k', cmap='Blues')
ax.set(title="Confusion Matrix Heatmap", xlabel="Predicted", ylabel="Actual",)
plt.show()
print('Total test digits:', cf_matrix.sum().item())
print('Predicted distribution:', cf_matrix.sum(0))
print('Actual distribution:', cf_matrix.sum(1))

