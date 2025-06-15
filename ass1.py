import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def intensity_level_reduction(image, levels):
    factor = 256 // levels
    reduced = (image // factor) * factor
    return reduced

def spatial_average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def rotate_image_by_angle(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def blockwise_average_downsample(image, block_size):
    h, w = image.shape
    downsampled = image.copy()
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y + block_size, x:x + block_size]
            avg = int(np.mean(block))
            downsampled[y:y + block_size, x:x + block_size] = avg
    return downsampled

def plot_and_save(title, image, filename, cmap='gray'):
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    image_path = "image.jpg"  
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(image_path)

    os.makedirs("outputs", exist_ok=True) 
    os.makedirs("outputs/intensity outputs", exist_ok=True)
    os.makedirs("outputs/average filters outputs", exist_ok=True) 
    os.makedirs("outputs/rotated outputs", exist_ok=True)  
    os.makedirs("outputs/blockwise outputs", exist_ok=True) 

    #  Intensity level reduction
    for level in [2, 4, 8, 16, 32, 64, 128, 256]:
        reduced_img = intensity_level_reduction(gray_img, level)
        filename = f"outputs/intensity outputs/intensity_{level}_levels.png"
        plot_and_save(f"Intensity {level} Levels", reduced_img, filename)

    #  Spatial average filtering
    for k in [3, 10, 20]:
        filtered_img = spatial_average_filter(gray_img, k)
        filename = f"outputs/average filters outputs/average_filter_{k}x{k}.png"
        plot_and_save(f"{k}x{k} Average Filter", filtered_img, filename)

    #  Rotate image by 45 and 90 degrees
    rotated_45 = rotate_image_by_angle(color_img, 45)
    rotated_90 = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
    plot_and_save("Rotated 45°", cv2.cvtColor(rotated_45, cv2.COLOR_BGR2RGB), "outputs/rotated outputs/rotated_45.png", cmap=None)
    plot_and_save("Rotated 90°", cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB), "outputs/rotated outputs/rotated_90.png", cmap=None)

    #  Block-wise average downsampling
    for b in [3, 5, 7]:
        downsampled_img = blockwise_average_downsample(gray_img, b)
        filename = f"outputs/blockwise outputs/blockwise_avg_{b}x{b}.png"
        plot_and_save(f"Blockwise Avg {b}x{b}", downsampled_img, filename)

    print(" All images processed and saved in the 'outputs/' folder.")

if __name__ == "__main__":
    main()
