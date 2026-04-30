# utils.py
import torch
import torchvision.transforms as transforms
from PIL import Image

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel size = 2, stride = 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) ,

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flattening
        x = self.fc_layers(x)

        return x


model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()
import torch
import torchvision.transforms as transforms
from PIL import Image

class_names = {
    0: "Fresh Apple",
    1: "Fresh Banana",
    2: "Fresh Orange",
    3: "Rotten Apple",
    4: "Rotten Banana",
    5: "Rotten Orange"
}


def predict_image(image):
    transform = transforms.Compose([
       transforms.Resize((32,32)),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_tensor = transform(image).unsqueeze(0)  
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

    
    return class_names[predicted.item()]
