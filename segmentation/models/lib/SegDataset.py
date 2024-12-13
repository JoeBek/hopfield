from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader



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


