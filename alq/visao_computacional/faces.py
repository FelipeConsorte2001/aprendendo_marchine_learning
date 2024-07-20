import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread("./../../img/workplace-1245776_1920.jpg")
# plt.imshow(imagem)
# plt.show()
detector_face = cv2.CascadeClassifier("./../../xml/haarcascade_frontalface_default.xml")
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("janela", imagem_cinza)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

deteccoes = detector_face.detectMultiScale(
    imagem_cinza, scaleFactor=1.3, minSize=(30, 30)
)
for x, y, l, a in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
cv2.imshow("janela", imagem)

cv2.waitKey(0)

cv2.destroyAllWindows()
