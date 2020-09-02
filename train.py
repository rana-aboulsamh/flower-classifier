# check that this works


import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from functions import get_data, train_model, test_model
from collections import OrderedDict

# Parser definition, get info from Command Line
parser = argparse.ArgumentParser()

# Get all components:
parser.add_argument('--data_dir', type = str, default = './flowers/', help = 'directory to the data with the images')
parser.add_argument('--arch', type = str, default = 'vgg16', help = 'choose architecture of CNN  pre-trained model')
parser.add_argument('--learning_rate', type = float, default = 0.001,  help = 'learning rate of the model')
parser.add_argument('--hidden_layers', type = int, default  = 1024,  help = 'hidden layers of CNN')
parser.add_argument('--dropout', type = float, default = 0.4, help = 'number as a float to avoid CNN overfitting')
parser.add_argument('--epochs', type = int, default = 2, help = 'number of cycles CNN goes through to train')
parser.add_argument('--gpu', type = str, default = 'gpu', help = "Choose between GPU or CPU")

# Parse arguments
args = parser.parse_args()
data_dir = args.data_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_layers = args.hidden_layers
dropout = args.dropout
epochs = args.epochs
gpu = args.gpu

def get_data(data_dir):
    
    # Define your transforms for the training, validation, and testing sets
    # Introduce randomization for your data to better train the Network. Images are rotated, cropped, and flipped
    train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform = valid_test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform = valid_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data


train_loader, valid_loader, test_loader, train_data, valid_data, test_data = get_data(data_dir) 
    
def choose_arch(arch = 'vgg16', hidden_layers = 4096, dropout = 0.4):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_layer = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_layer = 1024
    
    # Freeze parameters of pretrained model
    for param in model.parameters():
        param.requires_grad = False
   
    new_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_layer, hidden_layers, bias = True)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_layers, 102, bias = True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
        
    model.classifier = new_classifier 
    return model
        
model = choose_arch(arch, hidden_layers, dropout)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

def valid_train(model, valid_loader, criterion): 
    
    valid_loss = 0
    accuracy = 0
        
    for ii, (inputs, labels) in enumerate(valid_loader):
        if gpu == True:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
            
        # calculate probability
        probs = torch.exp(output)
            
        # calculate matches and how they relate to accuracy
        match = (labels.data == probs.max(dim=1)[1])
        accuracy += match.type(torch.FloatTensor).mean()
            
    return valid_loss, accuracy


def train_model(model, epochs, train_loader, valid_loader, test_loader, criterion, optimizer, gpu):
    steps = 0
    print_every = 40
    
    if gpu == True:
        model.to('cuda')
    
    for epoch in range(epochs):
        running_loss = 0
   
        for inputs, labels in train_loader:
            steps += 1
            
            if gpu == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
           
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            
            # perform backward pass
            loss.backward()
            
            # update parameters
            optimizer.step()
            
            # calculate training loss
            running_loss += loss.item()

            if steps % print_every == 0:
                # set model to evaluation mode
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = valid_train(model, valid_loader, criterion, gpu_mode) 

                print("Epoch: {}/{} | ".format(epoch+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(test_loader)), 
                      "Validation Accuracy: {:.4f}".format(accuracy/len(test_loader)))
                # reset running loss
                running_loss = 0

                # reset model back to training mode
                model.train()         

# call train_model:
train_model(model, epochs, train_loader, valid_loader, test_loader, criterion, optimizer, gpu)

def test_model(model, test_loader, gpu_mode):
    correct_matches = 0
    total_matches = 0
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
    
    with torch.no_grad():
       for data in test_loader:
           images, labels = data
            
           # Convert to CUDA
           if gpu_mode == True:
               images, labels = images.to('cuda'), labels.to('cuda')
           else:
               pass
            
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total_matches += labels.size(0)
           correct_matches += (predicted == labels).sum().item()
            
    print('Accuracy of Neural Network: %d%%' % (100 * correct_matches / total_matches))  

# call test_model
test_model(model)

# Save checkpoint:
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx
             }
torch.save(checkpoint, 'my_checkpoint.pth')     
    
