import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler 
from torch.utils.data import Subset, DataLoader		# create dataset, dataloader
import torchvision
from torchvision import datasets, transforms  # image data, transform to torch tensor format 
import sklearn
from sklearn.model_selection import train_test_split	# split dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from timeit import default_timer as timer
from datetime import timedelta
import argparse
from platform import python_version

def show_versions():
  print("python", python_version())
  print("torch", torch.__version__)
  print("torchvision", torchvision.__version__)
  print("matplotlib", matplotlib.__version__)
  print("seaborn", sns.__version__)
  print("sklearn", sklearn.__version__)

# Function to set device
def set_device(req_device):
  # Show cuda hardware
  if torch.cuda.is_available() == True:
    count = torch.cuda.device_count()
    print("GPU number of devices:", count)
    print(*["GPU device["+str(x)+"]="+torch.cuda.get_device_name(x) for x in range(count)], sep="\n")
    print("GPU current device:", torch.cuda.current_device())

  # Set device to CPU or GPU if available
  device = torch.device('cuda:0' if torch.cuda.is_available() and not (req_device == 'cpu') else 'cpu')
  print('Requested device:', req_device, ', using device:', device, '\n')
  return device

# Function to combine datasets into dict; split training dataset into train and validation
def train_val_split_test(train, val_size):
  train_idx, val_idx = train_test_split(list(range(len(train))), test_size=val_size)
  datasets = {}
  datasets['train'] = Subset(train, train_idx)
  datasets['val'] = Subset(train, val_idx)
  return datasets

# Show one image
def show_image(dataloader):
  images, labels = next(iter(dataloader))
  plt.imshow(images[0].view(28,28)) # images are tensors, need to reshape drop first index
  plt.xlabel(labels[0].item()) # labels are tensors, get value with item()
  plt.show()

# Preview dataset by batch
def show_images(dataloader, dataset_name):
  images, labels = next(iter(dataloader))
  plt.figure(figsize=(5,3))
  batch_size = images.shape[0]
  for i in range(batch_size):
    plt.subplot(int(batch_size/10),10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i].view(28,28), cmap=plt.cm.binary) # reshape tensor(1,28,28) to matplotlib shape(28,28) 
    plt.xlabel(labels[i].item()) # labels are tensors, get value with item()
    plt.suptitle('Batch of ' + str(batch_size) + ' images from ' + dataset_name, fontsize=16, y=.9)
  plt.show()

# Create a class to define the NN model.
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

# Define function to train the model
def train_model(model, criterion, optimizer, scheduler, epochs=25):
  start = timer()
  dataset_sizes = {x:len(datasets[x]) for x in ['train','val']}  

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

# Function to test model
def test_model(dataloader):
  correct, total = 0, 0
  print(device)
  with torch.no_grad():      # do not allocate memory for gradient calculations on model 
    for inputs, labels in dataloader:
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
  return (correct, total)

# Function to show a sample of prediction images, just one batch
# NOTE: images predicted on GPU if available, then moved to CPU for matplotlib plot
def show_prediction_images(dataloader, device):
  plt.figure(figsize=(16,8))
  test_images, test_labels = next(iter(dataloader))
  output = model(test_images.to(device).view(-1,784)).cpu()  
  batch_size = len(output)
  for i, img in enumerate(output):
    expected = test_labels[i]
    inferred = torch.argmax(img)
    cmap = plt.cm.binary if expected == inferred else plt.cm.autumn  
    plt.subplot(int(batch_size/20), 20, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].view(28,28), cmap=cmap)
    confidence = round(torch.max(torch.exp(img)).item()*100, 1)
    string = str(expected.item()) + "," + str(inferred.item()) + "," + str(confidence)
    plt.xlabel(string)
    plt.suptitle('Batch of ' + str(batch_size), fontsize=16, y=.9)
  plt.show()

# Function to show Confusion Matrix
def confusion_matrix(dataloader, num_classes, device):
  # Initialize confusion matrix
  class_names = [i for i in range(num_classes)]
  cf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.uint8)

  # Count expected and inferred results
  for images, labels in dataloader:
    output = model(images.to(device).view(-1,784)).cpu()  
    for i, img in enumerate(output):
      expected = labels[i]
      inferred = torch.argmax(img)
      cf_matrix[expected][inferred] += 1
 
  # Plot
  plt.figure(figsize=(10,6))
  ax = sns.heatmap(cf_matrix, annot=True, 
                 yticklabels=class_names, xticklabels=class_names, fmt='', 
                 linewidths=1, linecolor='k', cmap='Blues')
  ax.set(title="Confusion Matrix Heatmap", xlabel="Predicted", ylabel="Actual",)
  plt.show()
  print('Total test digits:', cf_matrix.sum().item())
  print('Predicted distribution:', cf_matrix.sum(0))
  print('Actual distribution:', cf_matrix.sum(1))

if __name__ == "__main__":
  show_versions()

  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-d","--device", help="device type(ex. cpu, cuda, cuda:0)")
  parser.add_argument("-t","--train", help="training mode", action="store_true")
  parser.add_argument("-p","--prod", help="production mode", action="store_true")
  args = parser.parse_args()
  
  # Set device to CPU or GPU
  device = set_device(args.device)

  # Download datasets
  train = datasets.MNIST('', train=True, download=True, 
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))
  test = datasets.MNIST('', train=False, download=True, 
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

  # Split train -> train+val datasets, put both in dictionary
  datasets = train_val_split_test(train, 0.25)
  print('Dataset sizes(train):', {x: len(datasets[x]) for x in ['train', 'val']})
  print('Dataset sizes(test):', len(test))
  
  if args.train:
    print("Training model...")
 
    # Create training dataloaders (iterable object over a dataset)
    dataloaders = {x:DataLoader(datasets[x], batch_size=10, shuffle=True, num_workers=4) for x in ['train','val']}
  
    # Show some images to verify dataset and dataloaders are functional
    show_image(dataloaders['train'])
    show_images(dataloaders['train'], 'train')

    # Train model
    model = Model().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    train_model(model, criterion, optimizer, scheduler, epochs=5)

    # Test model
    dataloader = DataLoader(test, batch_size=100, shuffle=False)
    correct, total = test_model(dataloader)
    print("Accuracy: ", round(correct/total, 3))
    show_prediction_images(dataloader, device)
    confusion_matrix(dataloader, 10, device)

    # Save model as PyTorch
    checkpoint = {'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

  if args.prod:
    print("Performing production mode inference...")

    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load model as PyTorch
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(checkpoint.keys())
    print(checkpoint['model_state_dict'].keys())
    print(checkpoint['optimizer_state_dict'].keys())

    # Test model
    dataloader = DataLoader(test, batch_size=100, shuffle=False)
    correct, total = test_model(dataloader)
    print("Accuracy: ", round(correct/total, 3))
    #show_prediction_images(dataloader, device)
    #confusion_matrix(dataloader, 10, device)
    
    # Measure time performance
    dataloader = DataLoader(test, batch_size=100, shuffle=False)
    count, fails = 0, 0
    for images, labels in dataloader:
      # move data to device
      images = images.to(device)
      labels = labels.to(device)
      output = model(images.to(device).view(-1,784)).cpu()
      count += len(output)
      for i, img in enumerate(output):
        expected = labels[i]
        inferred = torch.argmax(img)
        if expected != inferred:
          confidence = round(torch.max(torch.exp(img)).item()*100, 1)
          print("Inference error: " + str(expected.item()) + "," + str(inferred.item()) + "," + str(confidence))
          fails += 1
    print("Results: " + str(fails) + " errors in " + str(count) + " inferences (" + str(round(fails/count*100,2)) + "%)")
