import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction(image, prediction):
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
