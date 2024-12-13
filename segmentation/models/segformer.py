from lib.HopfieldModel import HopfieldModel
from transformers import SegformerForSemanticSegmentation


'''
Hopfield implentation for Segformer.
'''

class Segformer(HopfieldModel):

    def __init__(self):

        # define id to label conversions
        id2label = {0: 'background', 1: 'object'} 
        label2id = {'background': 0, 'object': 1} 

        model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        id2label=id2label,
        label2id=label2id)


        super().__init__(model)
    




if __name__ == "__main__":

    segformer = Segformer()

    segformer.inspect_modules(write=True)


