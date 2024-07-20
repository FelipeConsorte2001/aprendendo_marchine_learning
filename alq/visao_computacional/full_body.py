import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread("./../../img/pessoas.jpg")
# plt.imshow(imagem)
# plt.show()
detector = cv2.CascadeClassifier("./../../xml/fullbody.xml")
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# cv2.imshow("janela", imagem_cinza)
# cv2.waitKey(0)

# cv2.destroyAllWindows()
deteccoes = detector.detectMultiScale(imagem_cinza, minSize=(50, 50))
print(deteccoes)
for x, y, l, a in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
cv2.imshow("janela", imagem)

cv2.waitKey(0)

cv2.destroyAllWindows()
