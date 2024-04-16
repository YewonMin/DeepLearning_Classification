import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)

        # Extract label from filename
        label = int(img_name.split('_')[1].split('.')[0])

        return image, label

if __name__ == '__main__':
    # write test codes to verify your implementations
    dataset = MNIST(data_dir='path_to_your_MNIST_data_directory')
    print(len(dataset))
    img, label = dataset[0]
    print(img.shape, label)
