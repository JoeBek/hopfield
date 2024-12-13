from torchvision.models.segmentation import fcn_resnet50 
from lib.HopfieldModel import HopfieldModel, CityscapesDataset
from lib.SegDataset import SegDataset
from lib.Utils import get_path, visualize, iou, graph_iou
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class FCN(HopfieldModel):

    def __init__(self):

        model = fcn_resnet50(pretrained=True)
        super().__init__(model)
        # insert hopfield layer 
        self.model.backbone.layer4[2].add_module("hopfield", self.hopfield_layer)

 
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

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.backbone.layer4[2].hopfield.parameters():
            param.requires_grad = True
    


    def tune(self, dataloader, epochs=10, append="mkI"):
        self.model.train()
        self.freeze()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.backbone.layer4[2].hopfield.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            for batch in tqdm(dataloader):
                images, targets = batch[0].to(device), batch[1].to(device)
               # if targets.dim() == 1:
               #     targets = targets.unsqueeze(1)
                optimizer.zero_grad()
                outputs = self.model(images)
                logits = outputs['out']
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tqdm.write(f"Batch Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), get_path(f"../models/fcn_{append}.pt"))

        


        





if __name__ == "__main__":

    fcn = FCN()
    
    data = SegDataset(get_path(""))


    # fcn.tune(data.loader, epochs=10, append="mkI")


    """
    images, predictions = fcn.evaluate(data.loader)

    

    for i in range(len(images)):

        print(f"for image {i}")
        name = f"segment_{i + 40}_i.png"
        visualize(images[i], predictions[i], write=True, name=name)
    graph_iou(images, predictions, write=True, name="iou_40-49.png")
    """

        
    


    
        

