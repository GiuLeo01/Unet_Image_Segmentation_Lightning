from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torchvision import transforms

class SegmentationImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        self.img_labels = label_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_labels))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{idx+1}.png')
        image = read_image(img_path)
        label_path = os.path.join(self.img_labels, f'{idx+1}.png')
        label = read_image(label_path)[:][:][0].unsqueeze(0)
        label = label/ 255
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform_img = transforms.Compose([
    transforms.Resize((400, 400)),  # Ridimensiona l'immagine
    transforms.Lambda(lambda x: x.float())
])

transform_label = transforms.Compose([
    transforms.Resize((400, 400)),  # Ridimensiona l'immagine
    transforms.Lambda(lambda x: x.float()),
    transforms.Grayscale(num_output_channels=1)
])