from transformers import SegformerForSemanticSegmentation
import torch
from tqdm import tqdm
import os
from .Utils import get_path

"""        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            id2label=id2label,
            label2id=label2id
        )
"""



class HopfieldModel():

    def __init__(self, model):

        self.model = model



    def inspect_children(self, write=False):
        
        output = []
        for i, child in enumerate(self.model.children()):
            line = f"child {i} is: \n{child}\n"
            output.append(line)
            if not write:
                print(line)

        if write:
            path = get_path("dev/children_output.txt")
            with open(path, 'w') as file:
                file.writelines(output)

    def inspect_modules(self, write=False):
    
        output = []
        for i, module in enumerate(self.model.modules()):
            line = f"module {i} is: \n{module}\n"
            output.append(line)
            if not write:
                print(line)

        if write:
            path = get_path("dev/module_output.txt")
            with open(path, 'w') as file:
                file.writelines(output)


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
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1)
            images.append(image)
            predictions.append(prediction)


        return images, predictions



if __name__ == "__main__":

    id2label = {0: 'background', 1: 'object'} 
    label2id = {'background': 0, 'object': 1} 

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        id2label=id2label,
        label2id=label2id
    )
    hopfield_model = HopfieldModel(model)
    hopfield_model.inspect_children()
    hopfield_model.inspect_modules()



