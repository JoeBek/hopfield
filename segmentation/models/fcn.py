from torchvision.models.segmentation import fcn_resnet50 
from lib.HopfieldModel import HopfieldModel
from lib.SegDataset import SegDataset
from lib.Utils import get_path, visualize
import torch
import numpy as np
from tqdm import tqdm


class FCN(HopfieldModel):

    def __init__(self):

        model = fcn_resnet50(pretrained=True)
        super().__init__(model)

 
    # override evaluate because this output is slightly different

    def evaluate(self, dataloader, cutoff=10):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)


        images = []
        predictions = []

        for i,batch in enumerate(tqdm(dataloader, total=cutoff)):
            if i >= cutoff:
                break
            image = batch[0].to(device)
            with torch.no_grad():
                outputs = self.model(image)
            logits = outputs['out']
            prediction = torch.argmax(logits, dim=1)
            images.append(image)
            predictions.append(prediction)


        return images, predictions





if __name__ == "__main__":

    fcn = FCN()
    
    data = SegDataset(get_path("imagenet"))


    images, predictions = fcn.evaluate(data.loader)

    for i in range(len(images)):

        print(f"for image {i}")
        visualize(images[i], predictions[i])

