import argparse
import json
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image

# Get JSON file with all the flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Parser definition, get info from Command Line
parser = argparse.ArgumentParser()

# Get all components:
parser.add_argument('--image_dir', type = str, default = 'flowers/test/58/image_02672', help = 'directory with the test image')
parser.add_argument('--checkpoint_dir', type = str, default = 'my_checkpoint.pth', help = 'path to checckpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'top prediction results')
parser.add_argument('--names', type = str, default = 'cat_to_name.json',help = 'set path to naming category with an object')
parser.add_argument('--gpu', type = str, default = 'gpu', help = "Choose between GPU or CPU")

# Parse arguments:
args = parser.parse_args()
image_dir = args.image_dir
checkpoint_dir = args.checkpoint_dir
topk = args.topk
names = args.names
gpu = args.gpu

def load_checkpoint(checkpoint_dir):
    #load checkpoint path
    checkpoint = torch.load(checkpoint_dir)
    
    # download pre-trained network
    model = models.vgg16(pretrained=True)
    
    # freeze pre-trained network parameters
    for param in model.parameters(): param.requires_grad = False
        
    # load checkpoint content
    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# loads the model
load_model = load_checkpoint(checkpoint_dir)

def process_image(image_dir):
    # Process a PIL image for use in a PyTorch model
    pil_pic = Image.open(f'{image_dir}' + '.jpg')
    
    # Resize the image to meet network requirements
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                   ])
    pil_transform = transform(pil_pic)
    
    # Convert to a numpy array
    np_image = np.array(pil_transform)
    
    return np_image

def predict(image_dir, model, topk, gpu):
    
    # Implement the code to predict the class from an image file
    model = load_checkpoint(checkpoint_dir)
    if gpu == True:
        model.to('cuda')
    model.eval()
    
    # Get image
    torch_image = process_image(image_dir)
   
    # Convert image to a PyTorch Tensor
    torch_image_tensor = torch.from_numpy(torch_image).type(torch.cuda.FloatTensor)
    torch_image_tensor.unsqueeze_(0) # this needs to be added, to add a batch dimension to the input
    
    # Find probablities
    logps = model.forward(torch_image_tensor)
    linear_ps = torch.exp(logps)
    
    # Top 5 results
    top_ps, top_labels = linear_ps.topk(topk)
    
    top_ps = np.array(top_ps.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_flowers = [cat_to_name[label] for label in top_labels]
    
    return top_ps, top_labels, top_flowers

# Calculate prediction
top_ps, top_labels, top_flowers = predict(image_dir, model, topk, gpu)

print(f"This is a '{top_flowers[0]}' with a probability of {round(top_ps[0]*100,2)}% ")