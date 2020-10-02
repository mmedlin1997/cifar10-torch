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
train = datasets.MNIST('', train=True, download=True, 
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))
test = datasets.MNIST('', train=False, download=True, 
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))
print(train)

# split training dataset into training and validation dataset (dict)
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
train_idx, val_idx = train_test_split(list(range(len(train))), test_size=0.25)
datasets = {}
datasets['train'] = Subset(train, train_idx)
datasets['val'] = Subset(train, val_idx)

dataloaders = {x:torch.utils.data.DataLoader(datasets[x], 10, shuffle=True, num_workers=4) for x in ['train','val']}
x,y = next(iter(dataloaders['train']))
print(x.shape, y.shape)
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

# show one image
index = 0
plt.imshow(x[index].view(28,28))
plt.xlabel(y[index].item()) # labels are tensors, get value with item()
plt.show()

# Preview dataset
plt.figure(figsize=(5,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x[i].view(28,28), cmap=plt.cm.binary) # reshape tensor(1,28,28) to matplotlib shape(28,28) 
    plt.xlabel(y[i].item()) # labels are tensors, get value with item()
plt.show()

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

# Define fucntion to train the model
def train_model(model, criterion, optimizer, scheduler, epochs=25):
  start = timer()

  # init model state (save) and accuracy, in the end we return the best 
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode
    
      # init batch loss
      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        # move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # track history in train, not val
        torch.set_grad_enabled(phase == 'train')
        outputs = model(inputs.view(-1, 28*28))
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backpropagate and optimize in train, not val
        if phase == 'train':
          loss.backward()
          optimizer.step()
        
        # update batch statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      
      # step learning rate
      if phase == 'train':
        scheduler.step()

      # update epoch statistics
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
 
      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    print()
  
  # best statistics
  time_elapsed = timer() - start
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)

  return model

import torch.optim as optim
from torch.optim import lr_scheduler 
import copy 

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

train_model(model, criterion, optimizer, scheduler, epochs=5)

# test model
test_loader = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)
correct, total = 0, 0
print(device)
with torch.no_grad():      # do not allocate memory for gradient calculations on model 
    for inputs, labels in test_loader:
        # move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs.view(-1,784))
        #print(output)
        for idx, i in enumerate(output):
            #print(idx, i, torch.argmax(i), y[idx])
            if torch.argmax(i) == labels[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

# Save model as PyTorch
checkpoint = {'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

# Load model as PyTorch
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(checkpoint.keys())
print(checkpoint['model_state_dict'].keys())
print(checkpoint['optimizer_state_dict'].keys())

# Results
# Inspect one result
# NOTE: model is on GPU if avaiable, but CPU specified to matplotlib plots 
index = 5
expected = labels[index].cpu().item()
inferred = torch.argmax(model(inputs[index].view(-1,784))[0]).cpu().item()
plt.imshow(inputs[index].cpu().view(28,28))
plt.xlabel(inferred)
plt.show()

# Inspect batch
# NOTE: images predicted on GPU if available, then moved to CPU for matplotlib plot 
batch_size = 300
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
plt.figure(figsize=(10,6))
for test_images, test_labels in test_loader:
  output = model(test_images.to(device).view(-1,784)).cpu()  
  for i, img in enumerate(output):
    expected = test_labels[i]
    inferred = torch.argmax(img)
    cmap = plt.cm.binary if expected == inferred else plt.cm.autumn
    plt.subplot(batch_size/20, 20, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].view(28,28), cmap=cmap) 
    plt.xlabel(expected.item())
    plt.suptitle('Batch of ' + str(batch_size), fontsize=16, y=.9)
  break
plt.show()

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

print('Total test digits:', cf_matrix.sum().item())
print('Predicted distribution:', cf_matrix.sum(0))
print('Actual distribution:', cf_matrix.sum(1))

