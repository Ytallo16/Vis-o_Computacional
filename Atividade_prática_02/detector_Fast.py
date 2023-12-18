import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('img03.jpeg', cv2.IMREAD_GRAYSCALE)

# Inicializar o detector FAST
detector = cv2.FastFeatureDetector_create()

# Detectar pontos de interesse
pontos_chave = detector.detect(imagem, None)

# Desenhar os pontos de interesse na imagem
imagem_com_pontos = cv2.drawKeypoints(imagem, pontos_chave, None, color=(255, 0, 0))

# Exibir a imagem com os pontos de interesse
cv2.imshow('FAST Points', imagem_com_pontos)
cv2.waitKey(0)
cv2.destroyAllWindows()
