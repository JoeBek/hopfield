import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as func
import os

def visualize(image, prediction):
    # Convert the image and prediction to numpy arrays
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()

    # Denormalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Create a color map for the prediction
    cmap = plt.get_cmap('viridis')
    colored_prediction = cmap(prediction / prediction.max())

    # Display the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(colored_prediction)
    ax2.set_title('Segmentation Prediction')
    ax2.axis('off')
    plt.show()

def iou(image, prediction):
    # Convert the image to a binary mask
    image = torch.sum(image, dim=1) > 0  # Sum across the color channels and threshold
    image = image.float()  # Convert to float

    # Interpolate the prediction to match the image size
    prediction = func.interpolate(prediction.unsqueeze(1).float(), size=image.shape[1:], mode='nearest').squeeze(1)

    # Flatten the image and prediction
    image = image.view(image.size(0), -1)
    prediction = prediction.view(prediction.size(0), -1)

    # Calculate the intersection and union
    intersection = torch.sum(image * prediction, dim=1)
    union = torch.sum(image + prediction, dim=1) - intersection

    # Calculate the IoU
    iou = intersection / union
    return iou


def graph_iou(images, predictions):
    ious = []
    for image, prediction in zip(images, predictions):
        ious.append(iou(image, prediction))
    plt.scatter(range(len(ious)),ious)
    plt.xlabel('Image Index')
    plt.ylabel('IoU')
    plt.title('IoU vs. Image Index')
    plt.show()

def get_path(filename):

    # lib/
    libdir = os.path.dirname(os.path.abspath(__file__))
    # ../../data/dev
    datapath = f'../../data/{filename}'

    path = os.path.join(libdir, datapath)

    return path