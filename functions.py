import time
import torch
from torchvision import datasets, transforms, models

# Define a data loader function 
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

# Defining a valid training function to later train the network
def valid_train(model, valid_loader, criterion, gpu_mode): 
    valid_loss = 0
    accuracy = 0
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
        
    for ii, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
        start = time.time()
            
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
            
        # calculate probability
        probs = torch.exp(output)
            
        # calculate matches and how they relate to accuracy
        match = (labels.data == probs.max(dim=1)[1])
        accuracy += match.type(torch.FloatTensor).mean()
            
    return valid_loss, accuracy

# Function that actually trains the network
def train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, gpu_mode):
    steps = 0
    print_every = 40
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
    
    for epoch in range(epochs):
        running_loss = 0
   
        for inputs, labels in train_loader:
            steps += 1
            
            if gpu_mode == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass
            
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
                
    return model, optimizer

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