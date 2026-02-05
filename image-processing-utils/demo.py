from image_ops import ImageOps
import matplotlib.pyplot as plt

def run_demo():
    # 1. Load Data
    # (Using a placeholder, user should replace with their own image)
    try:
        img = ImageOps.load_image("sample_input.jpg")
    except:
        print("No input image found. Creating a dummy image.")
        img = np.random.rand(256, 256, 3)

    # 2. Run Pipeline
    print("Running Gaussian Blur...")
    kernel = ImageOps.gaussian_kernel(size=9, sigma=1.5)
    blurred = ImageOps.convolve_2d(img, kernel)

    print("Generating Gaussian Pyramid...")
    pyramid = ImageOps.gaussian_pyramid(img, levels=3)

    # 3. Visualize
    print("Displaying Results...")
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Blurred (Custom Convolution)")
    plt.imshow(blurred)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Pyramid Level 2 (Downsampled)")
    plt.imshow(pyramid[1])
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()