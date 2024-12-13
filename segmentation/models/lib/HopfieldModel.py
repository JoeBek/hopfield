from transformers import SegformerForSemanticSegmentation
import torch
from tqdm import tqdm
import os
from .Utils import get_path
from hflayers import HopfieldLayer

"""        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            id2label=id2label,
            label2id=label2id
        )
"""



class HopfieldModel():

    def __init__(self, model):

        self.model = model

        # each model can use the same hopfield layer for most of the flags
        # this layer is set for the FCN model, the sizes can be set in the inherited constructors
        self.hopfield_layer = HopfieldLayer(
                    input_size=2048,
                    hidden_size=512,
                    output_size=2048,
                    pattern_size=2048,
                    num_heads=8,
                    scaling=0.25,
                    update_steps_max=3,
                    update_steps_eps=1e-4,
                    normalize_stored_pattern=True,
                    normalize_stored_pattern_affine=True,
                    normalize_state_pattern=True,
                    normalize_state_pattern_affine=True,
                    normalize_pattern_projection=True,
                    normalize_pattern_projection_affine=True,
                    normalize_hopfield_space=False,
                    normalize_hopfield_space_affine=False,
                    stored_pattern_as_static=False
                )




    def inspect_children(self, write=False, prepend=""):
        
        output = []
        for i, child in enumerate(self.model.children()):
            line = f"child {i} is: \n{child}\n"
            output.append(line)
            if not write:
                print(line)

        if write:
            path = get_path(f"dev/{prepend}children_output.txt")
            with open(path, 'w') as file:
                file.writelines(output)

    def inspect_modules(self, write=False, prepend=""):
    
        output = []
        for i, module in enumerate(self.model.modules()):
            line = f"module {i} is: \n{module}\n"
            output.append(line)
            if not write:
                print(line)

        if write:
            path = get_path(f"dev/{prepend}module_output.txt")
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



