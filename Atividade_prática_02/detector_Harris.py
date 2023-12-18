import cv2
import numpy as np

# Carregar a imagem em escala de cinza
image = cv2.imread('img03.jpeg', cv2.IMREAD_GRAYSCALE)

# Parâmetros para o detector de Harris
block_size = 2
ksize = 3
k = 0.08
0
# Calcular derivadas parciais da imagem
Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize)
Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize)

# Calcular elementos da matriz de covariância
Ixx = Ix ** 2
Ixy = Ix * Iy
Iyy = Iy ** 2

# Aplicar a função de Harris
det_M = Ixx * Iyy - Ixy ** 2
trace_M = Ixx + Iyy
response = det_M - k * trace_M ** 2

# Normalizar a resposta para valores entre 0 e 255 com tipo de saída CV_8U
response_normalized = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Encontrar pontos de interesse usando cv2.goodFeaturesToTrack
corners = cv2.goodFeaturesToTrack(response_normalized, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Converter as coordenadas dos cantos para inteiros usando np.int_
corners = np.int_(corners)

# Extrair descritores (por exemplo, intensidade dos pixels) das regiões ao redor dos cantos
descriptors = []
for corner in corners:
    x, y = corner.ravel()
    
    # Verificar se a região ao redor do canto é grande o suficiente
    if 0 <= x - 5 < image.shape[1] and 0 <= y - 5 < image.shape[0] and 0 <= x + 5 < image.shape[1] and 0 <= y + 5 < image.shape[0]:
        patch = image[y - 5:y + 5, x - 5:x + 5]  # Região 10x10 ao redor do canto (ajuste conforme necessário)

        # Redimensionar a patch para um tamanho fixo (por exemplo, 10x10)
        patch = cv2.resize(patch, (10, 10))

        descriptor = patch.flatten()  # Usando intensidades dos pixels como descritores
        descriptors.append(descriptor)

# Converter para um array numpy
descriptors = np.array(descriptors)

# Exibir a imagem com os pontos de interesse de Harris
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(result_image, (x, y), 3, [0, 0, 255], -1)  # Pintar os pontos de interesse de vermelho

cv2.imshow('Harris Corners', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
