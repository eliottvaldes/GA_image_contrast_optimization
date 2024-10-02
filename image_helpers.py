import matplotlib.pyplot as plt
import numpy as np
import cv2


def read_image(path: str, image_height: int):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"La imagen en la ruta '{path}' no se encontró.")
    # Obtener el tamaño de la imagen y redimensio||nar si es necesario
    alto, ancho = img.shape[:2]
    #print(f'Medidas de imagen:\tAlto: {alto}, Ancho: {ancho}')
    if alto > image_height:
        proporcion = image_height / alto
        img = cv2.resize(img, (int(ancho * proporcion), image_height))
        alto, ancho = img.shape[:2]
        #print(f'Medidas de imagen ajustadas:\t Alto: {alto}, Ancho: {ancho}')
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    img = img.astype(np.float64) / 255.0  # Normalizar la imagen a [0, 1]
    return img

def show_results(img, description):
    plt.imshow(img, cmap="hot")
    plt.title(description)
    plt.show()


def plot_comparison(image, best_image, original_entropy, best_image_entropy, best_alfa, best_beta):
    best_image = (best_image * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original Image\nEntropy: {original_entropy}')
    axes[0].axis('off')

    # Best image
    axes[1].imshow(best_image, cmap='gray')
    axes[1].set_title(f'Best Image\nEntropy: {best_image_entropy}\n(alpha={best_alfa}, beta={best_beta})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image_path = './files/4360.png'
    image_height = 800
    image = read_image(image_path, image_height)
    # show image
    cv2.imshow('image',image)
    cv2.waitKey(0)