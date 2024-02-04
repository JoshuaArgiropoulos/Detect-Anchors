from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
from PIL import Image

class FormatData(Dataset): #constructor 
    def __init__(self, root_dir, mode='train', transform=None, target_size=(150, 150)):
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir, mode)
        self.target_size = target_size
        self.label_file = os.path.join(root_dir, mode, 'labels.txt').replace('\\', '/')

        # Reads labels
        self.labels = self._read_labels()

    # Helper function for getting label len()
    def __len__(self):
        return len(self.labels)

    # Retrieves the next partitioned image and corresponding label name
    def __getitem__(self, idx):
        
        img_name = os.path.normpath(os.path.join(self.image_folder, self.labels[idx][0]))
        image = cv2.imread(img_name)
        label = float(self.labels[idx][1])  # Car = 1.0, NoCar = 0.0

        image = cv2.resize(image, self.target_size)

        # Convert NumPy array to PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Transform the image
        if self.transform:
            image = self.transform(image)

        # Ensure label is a tensor with the correct shape
        label = int(self.labels[idx][1])  

        return image, label

    # Helper function to parse the labels file
    def _read_labels(self):
        labels = []
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                labels.append((parts[0], int(parts[1]), parts[2]))
        return labels
