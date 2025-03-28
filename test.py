from main import SiameseNet
from PIL import Image
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SNN_MODEL_TEST = SiameseNet().to(device)
SNN_MODEL_TEST.load_state_dict(torch.load('siamese_model.pth'))

SNN_MODEL_TEST.eval()

img1_path = 'datasets/logos/test/9.png'
img2_path = 'datasets/logos/test/10.png'

img1 = Image.open(img1_path).convert('RGB')
img2 = Image.open(img2_path).convert('RGB')

img1 = transform(img1).unsqueeze(0).to(device)
img2 = transform(img2).unsqueeze(0).to(device)

with torch.no_grad():
    output = SNN_MODEL_TEST(img1, img2)
    probability = torch.sigmoid(output).item()
    print(probability)
