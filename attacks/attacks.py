import torch


def fgsm_attack(images, gradient, epsilon=9/255):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = gradient.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = images + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image