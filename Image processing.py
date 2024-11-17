#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the OpenCV library
import cv2
import numpy as np
from IPython.display import display, Image
import matplotlib.pyplot as plt


# In[2]:


# Load a PNG image
image = cv2.imread("tulips.png")

# Display the PNG image
retval, buffer = cv2.imencode('.png', image)
img_encoded = buffer.tobytes()
display(Image(data=img_encoded, format='png'))




# In[3]:


# This line converts the loaded image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()


# In[4]:


# This line applies the binary threshold to the grayscale image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresh, cmap='gray')
plt.show()


# In[54]:


# Edge detection using Canny algorithm
edge_map_canny = cv2.Canny(gray, 100, 200)
retval, buffer = cv2.imencode('.png', edge_map_canny)
img_encoded = buffer.tobytes()
display(Image(data=img_encoded, format='png'))


# In[6]:


# Sobel Edge detection
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# Apply Sobel edge detection in the x and y directions
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert the gradient images to absolute values
sobel_x = np.absolute(sobel_x)
sobel_y = np.absolute(sobel_y)

# Combine the x and y gradient images to get the final edge map
edge_map_sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    
    
plt.imshow(edge_map_sobel, cmap='gray')

plt.axis('off')
plt.show()


# In[7]:


# Edge fusion
weight_canny=0.5
# Normalize edge maps to [0, 1]
canny_edge_map_norm = edge_map_canny.astype(np.float32) / 255.0
sobel_edge_map_norm = edge_map_sobel.astype(np.float32) / 255.0
# Perform edge fusion using weighted averaging
fused_edge_map = weight_canny * canny_edge_map_norm + (1 - weight_canny) * sobel_edge_map_norm
# Convert the fused edge map back to uint8 [0, 255]
fused_edge_map = (fused_edge_map * 255).astype(np.uint8)
plt.imshow(fused_edge_map,cmap='gray')





# In[51]:


# Adding Noise to the image

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Add salt noise
    num_salt = np.ceil(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    num_pepper = np.ceil(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

def add_gaussian_noise(image, mean=0, std=0.25):
    noisy_image = np.copy(image)
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image


# Load an example image
image = cv2.imread('tulips.png',cv2.IMREAD_GRAYSCALE)

# Add different types of noise
salt_and_pepper_noisy = add_salt_and_pepper_noise(image, salt_prob=0.0001, pepper_prob=0.0001)
gaussian_noisy = add_gaussian_noise(image)

# Display the original and noisy images

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(salt_and_pepper_noisy, cmap='gray')
plt.title('Salt and Pepper Noisy Image')

plt.subplot(133)
plt.imshow(gaussian_noisy, cmap='gray')
plt.title('Gaussian Noisy Image')
plt.show()




# In[52]:


# Calculate noise characteristics for the Salt and Pepper Noisy Image
def calculate_noise_characteristics(noisy_image, original_image):
    # Calculate the noise variance
    noise_variance = np.var(noisy_image - original_image)

    # Calculate the signal-to-noise ratio (SNR)
    signal_mean = np.mean(original_image)
    snr = 10 * np.log10(signal_mean ** 2 / noise_variance)

    # Calculate the peak signal-to-noise ratio (PSNR)
    mse = np.mean((original_image - noisy_image) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)

    return noise_variance, snr, psnr

# Calculate noise characteristics for the Salt and Pepper Noisy Image
noise_variance_salt_pepper, snr_salt_pepper, psnr_salt_pepper = calculate_noise_characteristics(salt_and_pepper_noisy, image)

# Calculate noise characteristics for the Gaussian Noisy Image
noise_variance_gaussian, snr_gaussian, psnr_gaussian = calculate_noise_characteristics(gaussian_noisy, image)




# Apply histogram equalization to the Salt and Pepper Noisy Image
equalized_salt_and_pepper = cv2.equalizeHist(salt_and_pepper_noisy)

# Apply histogram equalization to the Gaussian Noisy Image
equalized_gaussian = cv2.equalizeHist(gaussian_noisy)
# Display the histograms of the equalized images
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(equalized_salt_and_pepper.ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.7)
plt.title('Equalized Salt and Pepper Noise Histogram')

plt.subplot(132)
plt.hist(equalized_gaussian.ravel(), bins=256, range=(0, 256), density=True, color='g', alpha=0.7)
plt.title('    Equalized Gaussian Noise Histogram')

plt.show()




# Display the calculated noise characteristics
print("Salt and Pepper Noise Characteristics:")
print(f" - Noise Variance: {noise_variance_salt_pepper}")
print(f" - SNR: {snr_salt_pepper} dB")
print(f" - PSNR: {psnr_salt_pepper} dB")

print("\nGaussian Noise Characteristics:")
print(f" - Noise Variance: {noise_variance_gaussian}")
print(f" - SNR: {snr_gaussian} dB")
print(f" - PSNR: {psnr_gaussian} dB")







# In[53]:


# Denoising the Salt and Pepper Noisy Image using Median Filter
median_filtered_salt_pepper = cv2.medianBlur(salt_and_pepper_noisy, 5)  # Adjust the kernel size as needed

# Denoising the Gaussian Noisy Image using Median Filter
median_filtered_gaussian = cv2.medianBlur(gaussian_noisy, 5)  # Adjust the kernel size as needed

# Display the denoised images
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(median_filtered_salt_pepper, cmap='gray')
plt.title('Median Filter Salt and Pepper Denoised')

plt.subplot(122)
plt.imshow(median_filtered_gaussian, cmap='gray')
plt.title('Median Filter Gaussian Denoised')

plt.show()
# Denoising the Salt and Pepper Noisy Image using Gaussian Filter
gaussian_filtered_salt_pepper = cv2.GaussianBlur(salt_and_pepper_noisy, (5, 5), 0)  # Adjust the kernel size and std deviation as needed

# Denoising the Gaussian Noisy Image using Gaussian Filter
gaussian_filtered_gaussian = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)  # Adjust the kernel size and std deviation as neede

# Display the denoised images
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(gaussian_filtered_salt_pepper, cmap='gray')
plt.title('Gaussian Filter Salt and Pepper Denoised')

plt.subplot(122)
plt.imshow(gaussian_filtered_gaussian, cmap='gray')
plt.title('Gaussian Filter Gaussian Denoised')

plt.show()


# Define the Wiener filter kernel (you may need to customize this based on your specific noise characteristics)
# Define the Wiener filter function
def wiener_filter(noisy_image, kernel, noise_var):
    noisy_image_fft = np.fft.fft2(noisy_image)
    kernel_fft = np.fft.fft2(kernel, s=noisy_image.shape)
    
    wiener_output = np.fft.ifft2(noisy_image_fft * np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + noise_var))
    return np.abs(wiener_output).astype(np.uint8)

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

# Define the noise variance (you may need to estimate this from your noisy image)

# Apply Wiener filtering to the noisy images
wiener_filtered_salt_pepper = wiener_filter(salt_and_pepper_noisy, kernel, noise_variance_salt_pepper)
wiener_filtered_gaussian = wiener_filter(gaussian_noisy, kernel, noise_variance_gaussian)

# Display the original, noisy, and Wiener filtered images
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(132)
plt.imshow(salt_and_pepper_noisy, cmap='gray')
plt.title('Salt and Pepper Noisy Image')

plt.subplot(133)
plt.imshow(wiener_filtered_salt_pepper, cmap='gray')
plt.title('Wiener Filtered Salt and Pepper')

plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(gaussian_noisy, cmap='gray')
plt.title('Gaussian Noisy Image')

plt.subplot(133)
plt.imshow(wiener_filtered_gaussian, cmap='gray')
plt.title('Wiener Filtered Gaussian')

plt.show()


# In[ ]:




