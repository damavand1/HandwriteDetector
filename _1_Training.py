# _1_Training.py

#  https://www.youtube.com/watch?v=dNuUveTkCHo&list=PLGTnpzmoSsuHgWueAN72dVSqesigKMBJh
# https://pytorch.org/tutorials/

import torch
from _0_NN import NeuralNetwork,device

# import numpy as np

print(torch.__version__)

print("Using {device} device")

# Initializing a Tensor
# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
# //////////////////////////////////// Creating Tensor 1 (Directly from data)
data = [[1, 2],[3, 4]]
tensor1 = torch.tensor(data)
# //////////////////////////////////// Creating Tensor 2 (random)
shape = (2,3)
tensor2 = torch.rand(shape)
# //////////////////////////////////// Creating Tensor 3 (From another tensor)
x_ones = torch.ones_like(tensor1) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")   
# ////////////////////////////////

if torch.cuda.is_available():
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)
else:
    print ('NVIDIA GPU or Cuda is not available')

print(tensor1)
print(tensor1.dtype)
print(tensor1.shape)
print(tensor1.device)

# create a for loo

# data = [[1, 2],[3, 4]]   

# x_data = torch.tensor(data)   



# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)



# DevTeam: Create a Custom Dataset  

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
import torch.optim as optim


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for class_label in self.classes:
            class_path = os.path.join(self.root_dir, class_label)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                images.append((img_path, self.class_to_index[class_label]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    

# DevTeam: Create a Custom DataLoader

customDataset = CustomDataset(root_dir=os.path.abspath('0alphabet/0Data/0Images'), transform=transforms.Compose([transforms.ToTensor()]))    
trainDataLoader = DataLoader(dataset=customDataset, batch_size=32, shuffle=True)



# Define a simple neural network

model = NeuralNetwork().to(device)
print(model)



      

    
    
    






    



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)





num_epochs = 10  # Adjust the number of epochs as needed

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in trainDataLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainDataLoader)}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Save the trained model and class-to-index mapping
# torch.save({
#      'model_state_dict': model.state_dict(),
#      'class_to_index': customDataset.class_to_index
#  }, 'trained_model_with_mapping.pth')
