import cv2

# Carregar a imagem
image = cv2.imread('img03.jpeg', cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem.")
else:
    # Inicializar o detector SIFT
    sift = cv2.SIFT_create()

    # Encontrar pontos de interesse e descritores usando SIFT
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Desenhar os pontos de interesse na imagem
    result_image = cv2.drawKeypoints(image, keypoints, None)

    # Exibir a imagem com os pontos de interesse
    cv2.imshow('SIFT Keypoints', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
