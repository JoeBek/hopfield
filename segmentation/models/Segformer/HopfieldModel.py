from transformers import SegformerForSemanticSegmentation
import torch
from tqdm import tqdm

"""        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            id2label=id2label,
            label2id=label2id
        )
"""



class HopfieldModel():

    def __init__(self, model):

        self.model = model



    def inspect_children(self):
        
        for i, child in enumerate(self.model.children()):
            print(f"child {i} is: ")
            print(child)

    def inspect_modules(self):
    
        for i, module in enumerate(self.model.modules()):
            print(f"module {i} is: ")
            print(module)

    def evaluate(self, dataloader, cutoff=10):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)


        for i,image in enumerate(tqdm(dataloader, total=cutoff)):
            if i >= cutoff:
                break
            image = image.to(device)
            with torch.no_grad():
                outputs = self.model(image)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            print(predictions)





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




        


