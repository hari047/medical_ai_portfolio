import numpy as np
from PIL import Image
from typing import List, Tuple, Union

class ImageOps:
    """
    A foundational image processing library built from scratch using NumPy.
    
    Capabilities:
    - 2D Convolution (Custom implementation, no SciPy)
    - Gaussian & Laplacian Kernel Generation
    - Image Pyramids (Gaussian)
    - Resizing (Nearest Neighbor) & Normalization
    """

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Loads an image and normalizes it to a 0-1 numpy array."""
        try:
            img = Image.open(path)
            # Convert to float and normalize to [0, 1]
            return np.asarray(img).astype('float32') / 255.0
        except Exception as e:
            raise IOError(f"Failed to load image at {path}: {e}")

    @staticmethod
    def save_image(image: np.ndarray, path: str):
        """Saves a normalized numpy array (0-1) as an image file."""
        # Clip to safe range and convert back to 0-255 uint8
        img_uint8 = (np.clip(image, 0, 1) * 255).astype('uint8')
        Image.fromarray(img_uint8).save(path)

    @staticmethod
    def to_greyscale(image: np.ndarray) -> np.ndarray:
        """Converts RGB image to Greyscale using luminance weights."""
        if len(image.shape) == 2:
            return image # Already greyscale
        # Standard luminance weights: 0.299R + 0.587G + 0.114B
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """
        Generates a 2D Gaussian kernel from scratch.
        """
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel) # Normalize

    @staticmethod
    def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Applies 2D convolution. Supports both Grayscale (2D) and RGB (3D) inputs.
        """
        kernel_h, kernel_w = kernel.shape
        pad_h, pad_w = kernel_h // 2, kernel_w // 2

        # Handle RGB images by convolving each channel separately
        if len(image.shape) == 3:
            channels = []
            for c in range(image.shape[2]):
                channels.append(ImageOps.convolve_2d(image[:, :, c], kernel))
            return np.stack(channels, axis=2)

        # Pad image (Reflect padding reduces edge artifacts)
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        output = np.zeros_like(image)

        # Vectorized implementation for speed (or nested loops for 'from scratch' demo)
        # Using a sliding window approach here for clarity:
        for i in range(kernel_h):
            for j in range(kernel_w):
                output += kernel[i, j] * padded_image[i:i+image.shape[0], j:j+image.shape[1]]
        
        return output

    @staticmethod
    def resize(image: np.ndarray, scale: float) -> np.ndarray:
        """
        Resizes an image using Nearest Neighbor interpolation.
        """
        if scale <= 0:
            raise ValueError("Scale must be positive")
            
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Grid generation
        row_indices = (np.arange(new_h) / scale).astype(int)
        col_indices = (np.arange(new_w) / scale).astype(int)
        
        # Advanced Numpy indexing to construct the new image
        if len(image.shape) == 3:
            return image[row_indices[:, None], col_indices]
        return image[np.ix_(row_indices, col_indices)]

    @staticmethod
    def gaussian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
        """
        Constructs a Gaussian Pyramid.
        
        Process:
        1. Smooth image (Gaussian Blur)
        2. Downsample (Reduce size by half)
        3. Repeat
        """
        pyramid = [image]
        kernel = ImageOps.gaussian_kernel(size=5, sigma=1.0)
        
        current_img = image
        for _ in range(levels - 1):
            # Blur
            blurred = ImageOps.convolve_2d(current_img, kernel)
            # Downsample (Resize by 0.5)
            downsampled = ImageOps.resize(blurred, scale=0.5)
            
            pyramid.append(downsampled)
            current_img = downsampled
            
        return pyramid