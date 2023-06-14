import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.transforms import functional

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Show Function for image tensor
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def pre_image(image_path,model):
   transform_norm = transforms.Compose([
       transforms.Resize((70,70)),
       transforms.ToTensor()
       ])

   # Read Image
   img = Image.open(image_path).convert('L')
   img = ImageOps.invert(img).convert("RGB")
   img_normalized = transform_norm(img).float()
   show(img_normalized)
   img_normalized = img_normalized.unsqueeze_(0)
   img_normalized = img_normalized.to(device)

   print("Shape Image:",img_normalized.shape)
   with torch.no_grad(): # Predict Image
      model.eval()
      output =model(img_normalized)
     # print(output)
      index = output.data.cpu().numpy().argmax()
      classes = ["Ankle Bot", "Bag", "Coat", "Dress", "Hat","Pullover", "Sandal", "Shirt", "Sneaker", "T-shirt/Top", "Trouser"]
      class_name = classes[index]
      return class_name