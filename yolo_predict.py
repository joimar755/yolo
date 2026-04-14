import cv2
from ultralytics import YOLO
import os

# Ruta base del proyecto
base_path = os.path.dirname(os.path.abspath(__file__))

# Ruta de la imagen
ruta_completa = os.path.join(base_path, "img", "1.jpg")

# Cargar modelo
model = YOLO("best.pt")

# Leer imagen
img = cv2.imread(ruta_completa)

if img is None:
    print("Error: no se pudo cargar la imagen")
    exit()

# Ejecutar detección
results = model(img)

# Mostrar resultados
annotated = results[0].plot()
cv2.imshow("Detecciones", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()