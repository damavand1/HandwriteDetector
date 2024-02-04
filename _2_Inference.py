# _2_Inference.py

import os
import torch
from torchvision import transforms
from PIL import Image
from _0_NN import NeuralNetwork,device



################################## Loading the Model for Inference
# Instantiate the model
model = NeuralNetwork()

# Load the trained weights
model.load_state_dict(torch.load('trained_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode


################################## Inference on New Data
# Assume you have a new image for inference
# new_image_path = root_dir=os.path.abspath('0alphabet/0Data')+'/test/id_7393_label_1.png'
# new_image_path = root_dir=os.path.abspath('0alphabet/0Data')+'/test/id_6064_label_2.png'
# new_image_path = root_dir=os.path.abspath('0alphabet/0Data')+'/test/id_245_label_3.png'
new_image_path = root_dir=os.path.abspath('0alphabet/0Data')+'/test/id_245_label_3.png'




new_image = Image.open(new_image_path).convert("RGB")
new_image = transforms.ToTensor()(new_image).unsqueeze(0).to(device)


# Make predictions
with torch.no_grad():
    model.eval()
    output = model(new_image)

# Get predicted class
_, predicted_class = torch.max(output, 1)

print(predicted_class)

# Manually specify class names based on your dataset
class_names = ["ا", "ب", "ت", ...]  # Add the actual class names here

print(f'The model predicts the image belongs to class: {predicted_class.item()}')


predicted_class_index = predicted_class.item()
predicted_class_name = class_names[predicted_class_index]
print(f'The model predicts the image belongs to class: {predicted_class_index} ({predicted_class_name})')