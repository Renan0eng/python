import cv2
import numpy as np
import os

# Obtém o diretório do script Python
diretorio_script = os.path.dirname(__file__)

# Caminho para a pasta contendo a imagem
pasta_imagens = f"{diretorio_script}/images"

# Lista todos os arquivos da pasta de imagens
try:
    imagens_arquivos = os.listdir(pasta_imagens)
except FileNotFoundError:
    raise FileNotFoundError(f"A pasta de imagens '{pasta_imagens}' não foi encontrada.")

def preProcessing(img):
    imgPre0 = cv2.Canny(img, 130, 130)
    # mostra a imagem com o filtro de canny
    cv2.imshow("Canny1", imgPre0) 

    return imgPre0

# Mostra todas as imagens da pasta
for i, imagem_arquivo in enumerate(imagens_arquivos):
    print(f"{i}: {imagem_arquivo}")

# Pergunta ao usuário qual imagem ele deseja visualizar
imagem_escolhida = int(input("Digite o número da imagem que deseja visualizar: "))
if imagem_escolhida < 0 or imagem_escolhida >= len(imagens_arquivos):
    raise ValueError(f"O valor {imagem_escolhida} é inválido.")

# Obtém o caminho completo da imagem escolhida
imagem_arquivo = f"{pasta_imagens}/{imagens_arquivos[imagem_escolhida]}"
print(f"Imagem escolhida: {imagem_arquivo}")

# Leitura da imagem escolhida
img = cv2.imread(imagem_arquivo)
if img is None:
    raise FileNotFoundError(f"Não foi possível ler a imagem: {imagem_arquivo}")
if img.size == 0:
    raise ValueError("A imagem está vazia.")
if img.shape[:2] == (0, 0):
    raise ValueError("A imagem tem dimensões inválidas.")

# Redimensiona a imagem para um tamanho padrão (640x480)
img = cv2.resize(img, (640, 480))

# Aplica o pré-processamento para realçar bordas
imgThres = preProcessing(img)

# Encontra contornos na imagem processada
countors, h1 = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Contagem de contornos com área maior que 100
count_up2000 = 0

# Criando a pasta roi, se não existir
roi_folder = f"{diretorio_script}/roi"
os.makedirs(roi_folder, exist_ok=True)

# Itera sobre os contornos encontrados
for idx, cnt in enumerate(countors):
    area = cv2.contourArea(cnt)
    if area > 100:
        count_up2000 += 1
        cv2.drawContours(img, cnt, -1, (255, 0, 0), 1)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        print(len(approx))
        x, y, w, h = cv2.boundingRect(approx)
        
        # Salva a imagem com um nome único usando o índice do contorno
        imgROI = img[y:y + h, x:x + w]
        cv2.imwrite(f"{roi_folder}/{idx}.png", imgROI)
        
# Exibindo a quantidade de objetos contornados na tela
cv2.putText(img, "Objects: " + str(count_up2000), (10, 470), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

# Exibe a imagem com os contornos
cv2.imshow("Contours", img)

# Aguarda uma tecla ser pressionada
cv2.waitKey(0)
