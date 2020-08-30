import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from functions import get_data, train_model, test_model
from collections import OrderedDict

# Parser definiton
parser = argparse.ArgumentParser(description = 'Training a Neural Network to identify images')

# checkpoint directory to parser
parser.add_argument('--data_dir', type = str, dest = 'data_directory', help = 'checkpoint directory')

# Parser architechture
parser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN model')

# Parser hyperparameters
parser.add_argument('--learning_rate', type = float, dest = 'lr', default = 0.001,  help = 'learning rate of the model')
parser.add_argument('--epochs', type = int, dest = 'epochs', default = 2, help = 'number of cycles CNN goes through to train')
parser.add_argument('--dropout', type = float, dest = 'dropout', default = 0.4, help = 'number as a float to avoid CNN overfitting')
parser.add_argument('--hidden_layers', type = int, dest = 'hidden_layers', default  = 512,  help = 'hidden layers of CNN')
    
# GPU option for easier training
parser.add_argument('--gpu', action = "store_true", dest = 'gpu', help = "Use GPU and CUDA for easier calculations and faster training")
    
# Parse arguments
args = parser.parse_args()
data_dir = args.data_directory
hidden_layers = args.hidden_layers
dropout = args.dropout
lr = args.lr
epochs = args.epochs
gpu_mode = args.gpu


# Get and process images
train_loader, valid_loader, test_loader, train_data, valid_data, test_data = get_data(data_dir)
arch = args.arch

if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_layers)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_layers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
elif arch == 'alexnet':
    model = models.alexnet(pretrained = True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(4096, hidden_layers)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(dropout)),
        ('fc3', nn.Linear(hidden_layers, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
# Freeze parameters for pretrained models
for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier

# Set up optimizer and criterion
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Train the model
model, optimizer = train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, gpu_mode)

# Test how accurate the model is
test_model(model, test_loader, gpu_mode)

# Save model checkpoint so we don't lose progress
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'arch': arch,
              'class_to_idx': train_data.class_to_idx
             }
torch.save(checkpoint, 'my_checkpoint.pth')