from PIL import Image
import numpy as np
import zipfile
import os
import cv2

path = "./../../img/yalefaces.zip"
zip_object = zipfile.ZipFile(file=path, mode="r")
zip_object.extractall("./")
zip_object.close()


def dados_imagem():
    caminhos = [
        os.path.join("./yalefaces/train", f) for f in os.listdir("./yalefaces/train")
    ]
    faces = []
    ids = []
    for caminho in caminhos:
        imagem = Image.open(caminho).convert("L")
        imagem_np = np.array(imagem, "uint8")
        id = int(os.path.split(caminho)[1].split(".")[0].replace("subject", ""))
        ids.append(id)
        faces.append(imagem_np)
    return np.array(ids), faces


ids, faces = dados_imagem()
print(ids, faces)
lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, ids)
lbph.write("classificadorLBPH.yml")

lbph.read("classificadorLBPH.yml")
imagem_teste = "./yalefaces/test/subject10.sad.gif"
imagem = Image.open(imagem_teste).convert("L")
imagem_np = np.array(imagem, "uint8")
print(imagem_np)
id_previsto, _ = lbph.predict(imagem_np)
print(id_previsto)
id_correto = int(os.path.split(imagem_teste)[1].split(".")[0].replace("subject", ""))
print(id_correto)


cv2.putText(
    imagem_np,
    "P:" + str(id_previsto),
    (30, 30 + 30),
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    1,
    (0, 255, 0),
)
cv2.putText(
    imagem_np,
    "C:" + str(id_correto),
    (30, 50 + 30),
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    1,
    (0, 255, 0),
)
cv2.imshow("janela", imagem_np)

cv2.waitKey(0)

cv2.destroyAllWindows()
