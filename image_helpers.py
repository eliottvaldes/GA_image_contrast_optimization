import numpy as np
import cv2

def read_image(path: str, image_height: int):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"The image {path} was not found")
    height, width = img.shape[:2]
    if height > image_height:
        proportion = image_height / height
        img = cv2.resize(img, (int(width * proportion), image_height))
        height, width = img.shape[:2]
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.0
    return img


if __name__ == '__main__':
    # Test the read_image function
    image_path = './assets/4360.png'
    image_height = 80
    image = read_image(image_path, image_height)
    cv2.imshow('image', image)
    cv2.waitKey(0) 