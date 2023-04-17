import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# Converta a imagem para escala de cinza
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Aplique um filtro Gaussiano para suavizar a imagem e remover ruído
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplique uma operação de limiarização para converter a imagem em preto e branco
thresh = cv2.threshold(blur, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Aplique uma operação de abertura para remover pequenos objetos indesejados
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# Encontre os contornos dos objetos presentes na imagem
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Classifique os contornos por área para encontrar o maior objeto, que será o bovino
if len(contours) > 0:
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  bovino_contour = contours[0]
else:
  bovino_contour = None
# Calcule a área do bovino
if bovino_contour is not None:
  area = cv2.contourArea(bovino_contour)

  # Calcule o peso do bovino com base na área
  # (este é apenas um exemplo básico e não deve ser usado para fins reais de pesagem)
  peso = area * 2.5
else:
  area = None
  peso = None
