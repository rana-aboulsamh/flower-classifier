import argparse
import json
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image

# Parser definiton
parser = argparse.ArgumentParser(description = 'Neural netwwork predicts and returns the top results for an image')

# image directories
parser.add_argument('--img_dir', type = str, default = 'flowers/test/58/image_02672', dest = 'image_directory', help = 'directory to the image')
parser.add_argument('--check_dir', type = str, dest = 'checkpt_directory', default = 'my_checkpoint.pth', help = 'directory to the image')

# Top results
parser.add_argument('--topk', type = int, dest = 'results', default = 5, help = 'top prediction results')

# Name mapping
parser.add_argument('--names', type = str, dest = 'names', default = 'cat_to_name.json',help = 'set path to naming category with an object')

# GPU option for easier training
parser.add_argument('--gpu', action = "store_true", dest = 'gpu', help = "Use GPU and CUDA for easier calculations and faster training")

# Parse arguments
args = parser.parse_args()
img_dir = args.image_directory
gpu_mode = args.gpu
names = args.names
topk = args.results
checkpt = args.checkpt_directory


# Load checkpoint to rebuild the model
checkpoint = torch.load(checkpt)
if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained = True)
elif checkpoint['arch'] == 'alexnet':
    model = models.alexnet(pretrained = True)
    
model.state_dict (checkpoint['state_dict'])
model.classifier = checkpoint['classifier']
model.class_to_idx = checkpoint['class_to_idx']

# Freeze parameters 
for param in model.parameters():
    param.requires_grad = False

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
   
def process_image(img_dir):
    #Process a PIL image for use in a PyTorch model
    pil_pic = Image.open(f'{img_dir}' + '.jpg')
    
    # Resize the image to meet network requirements
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                   ])
    
    pil_transform = transform(pil_pic)
    
    # Convert PyTorch Tensor to a NumPy array
    np_image = np.array(pil_transform)
    
    # Convert from NumPy array to PyTorch Tensor
    torch_image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
    
    # This needs to be added, to add a batch dimension to the input
    processed_img = torch_image_tensor.unsqueeze_(0)

    return processed_img
    

def predict(img_dir, model, topk, gpu_mode):

    # Implement the code to predict the class from an image file
    
    # Get image
    torch_image = process_image(img_dir)
    
    # switch to GPU mode
    if gpu_mode == True:
        model.to('cuda')
    else:
        model.to('cpu')
    
    # Convert image to CUDA
    if gpu_mode == True:
        torch_image.to('cuda')
    else:
        pass
    
    # Find probablities
    logps = model.forward(torch_image)
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
probs, labels = predict(img_dir, model, topk, gpu_mode)

print(top_ps)
print(top_labels)
print(top_flowers)








