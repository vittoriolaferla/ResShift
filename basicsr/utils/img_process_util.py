import cv2
import numpy as np
import torch
from torch.nn import functional as F



def filter2D(img, kernel):
    """
    PyTorch version of cv2.filter2D

    Args:
        img (Tensor): Input image tensor of shape (b, c, h, w)
        kernel (Tensor): Kernel tensor of shape (b, k, k)

    Returns:
        Tensor: Filtered image tensor of shape (b, c, h, w)
    """
    k = kernel.size(-1)
    assert k % 2 == 1, "Kernel size should be odd for symmetric padding."

    b, c, h, w = img.size()
    ph, pw = h, w  # Assuming ph and pw refer to height and width

    padding = k // 2  # For 'same' padding

    if kernel.size(0) == 1:
        # Apply the same kernel to all batch images and channels
        img_reshaped = img.reshape(b * c, 1, ph, pw)  # Shape: (b*c, 1, h, w)
        kernel_reshaped = kernel.reshape(1, 1, k, k)  # Shape: (1, 1, k, k)
        conv_out = F.conv2d(img_reshaped, kernel_reshaped, padding=padding)
        filtered_img = conv_out.reshape(b, c, h, w)
     #   print(f"Filtered image shape (same kernel): {filtered_img.shape}")  # Debug
        return filtered_img
    else:
        # Apply different kernels for each image in the batch to each channel
        assert kernel.size(0) == b, "Number of kernels must match batch size."
        
        # Reshape kernel to (b, 1, k, k)
        kernel_reshaped = kernel.reshape(b, 1, k, k)  # Shape: (b, 1, k, k)
        # Repeat kernel across channels to get (b*c, 1, k, k)
        kernel_repeated = kernel_reshaped.repeat(1, c, 1, 1).reshape(b * c, 1, k, k)  # Shape: (b*c, 1, k, k)
        
        # Reshape img to (1, b*c, h, w)
        img_reshaped = img.reshape(1, b * c, ph, pw)  # Shape: (1, b*c, h, w)
        
        # Debugging statements
      #  print(f"Kernel repeated shape: {kernel_repeated.shape}")  # Expected: [b*c, 1, k, k]
       # print(f"Image reshaped shape: {img_reshaped.shape}")      # Expected: [1, b*c, h, w]")
        
        # Perform grouped convolution
        conv_out = F.conv2d(img_reshaped, kernel_repeated, padding=padding, groups=b * c)
        # Reshape back to (b, c, h, w)
        filtered_img = conv_out.reshape(b, c, h, w)
        
        # Debugging statement
      #  print(f"Filtered image shape (different kernels): {filtered_img.shape}")  # Debug
        
        return filtered_img


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img
