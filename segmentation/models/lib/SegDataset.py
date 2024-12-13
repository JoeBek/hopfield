from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os



"""
dataset loader for segmentation models


transforms.Resize((512, 512)),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

"""
class SegDataset(Dataset):
    def __init__(self, path):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset = datasets.ImageFolder(path, transform=self.transform)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True)



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        image = self.transform(image)
        return image

    def get_image(self, index):

        return self.dataset[index][0]

class CityscapesDataset(Dataset):
    def __init__(self, raw_images_dir, ground_truth_dir, transform=None):
        self.raw_images_dir = raw_images_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(raw_images_dir))
        self.gt_filenames = sorted(os.listdir(ground_truth_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.raw_images_dir, self.image_filenames[idx])
        gt_image_path = os.path.join(self.ground_truth_dir, self.gt_filenames[idx])

        raw_image = Image.open(raw_image_path).convert("RGB")
        gt_image = Image.open(gt_image_path).convert("L")  

        if self.transform:
            raw_image = self.transform(raw_image)
            gt_image = self.transform(gt_image)

        return raw_image, gt_image
    

if __name__ == "__main__":
    dataset = SegDataset("/home/joe/vt/ml/hopfield-layers/segmentation/data/imagenet")

    image = dataset.get_image(0)
    print(image)
    print(image.shape)
    print(image.dtype)
    print(image.min())
    print(image.max())
    print(image.mean())
    print(image.std())


