# Image-processing

README: Image Processing with OpenCV and Python
===============================================

Overview
--------
This script demonstrates various image processing techniques using Python and OpenCV. The primary focus is on loading, visualizing, and manipulating images through edge detection, noise addition, and noise removal methods. The code is structured to work in a Jupyter Notebook environment.

Features
--------
1. **Image Loading and Display**:
   - Load an image (`tulips.png`) and display it in its original format.
   - Convert the image to grayscale for further processing.

2. **Edge Detection**:
   - **Canny Edge Detection**: Identifies edges based on intensity gradients.
   - **Sobel Edge Detection**: Computes edges in x and y directions and combines them.

3. **Edge Fusion**:
   - Merges edge maps from Canny and Sobel algorithms using weighted averaging.

4. **Noise Addition**:
   - **Salt and Pepper Noise**: Introduces random white (salt) and black (pepper) pixels.
   - **Gaussian Noise**: Adds random noise with a Gaussian distribution.

5. **Noise Analysis**:
   - Calculates noise variance, Signal-to-Noise Ratio (SNR), and Peak Signal-to-Noise Ratio (PSNR).

6. **Noise Reduction**:
   - **Median Filtering**: Removes noise using a median filter.
   - **Gaussian Filtering**: Smoothens images while retaining edges.
   - **Wiener Filtering**: Reduces noise using frequency-domain filtering.

7. **Histogram Equalization**:
   - Enhances the contrast of images, particularly those with noise.

8. **Visualization**:
   - Side-by-side comparison of original, noisy, and denoised images.
   - Displaying edge maps, histograms, and noise characteristics.

Requirements
------------
- Python 3.x
- Libraries:
  - `opencv-python` (`cv2`)
  - `numpy`
  - `matplotlib`
  - `IPython`

Install dependencies using:
```
pip install opencv-python numpy matplotlib ipython
```

Usage Instructions
------------------
1. **Run the Code**:
   - Save the script in a Python-compatible environment (e.g., Jupyter Notebook).
   - Ensure `tulips.png` is in the same directory as the script.

2. **Customize Parameters**:
   - Modify noise levels, edge detection thresholds, and filtering parameters to experiment with different effects.

3. **Output**:
   - The script generates and displays images for each processing step.
   - Noise characteristics and metrics are printed in the console.

Key Functions
-------------
- **Noise Addition**:
  ```python
  add_salt_and_pepper_noise(image, salt_prob, pepper_prob)
  add_gaussian_noise(image, mean=0, std=0.25)
  ```
- **Edge Detection**:
  ```python
  cv2.Canny(image, threshold1, threshold2)
  cv2.Sobel(image, ddepth, dx, dy, ksize)
  ```
- **Noise Filtering**:
  ```python
  cv2.medianBlur(image, ksize)
  cv2.GaussianBlur(image, ksize, sigmaX)
  wiener_filter(noisy_image, kernel, noise_var)
  ```

Acknowledgments
---------------
This script showcases fundamental image processing concepts suitable for educational and experimental purposes. For advanced applications, further refinement and optimization may be required.
