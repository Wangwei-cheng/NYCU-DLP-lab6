import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import torchvision.transforms as transforms

class ICLEVRDataset(Dataset):
    def __init__(self, img_dir, train_path, objects_path, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            train_path (string): Path to the training json file with annotations.
            objects_path (string): Path to the objects.json for mapping.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        with open(train_path, 'r') as f:
            self.data = json.load(f)
        with open(objects_path, 'r') as f:
            self.objects = json.load(f)
        
        self.filenames = list(self.data.keys())
        self.num_classes = len(self.objects)
        
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle error as needed
            image = Image.new('RGB', (64, 64), (0, 0, 0))
            
        image = self.transform(image)
        
        labels = self.data[filename]
        multi_hot = torch.zeros(self.num_classes)
        for label in labels:
            index = self.objects[label]
            multi_hot[index] = 1
            
        return image, multi_hot

def get_test_conditions(test_path, objects_path):
    """
    Returns a tensor of multi-hot encoded conditions from test.json or new_test.json.
    """
    with open(test_path, 'r') as f:
        data = json.load(f)
    with open(objects_path, 'r') as f:
        objects = json.load(f)
    
    num_classes = len(objects)
    conditions = []
    for labels in data:
        multi_hot = torch.zeros(num_classes)
        for label in labels:
            index = objects[label]
            multi_hot[index] = 1
        conditions.append(multi_hot)
    
    return torch.stack(conditions)
