import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Obtém o diretório do script Python
diretorio_script = os.path.dirname(__file__)

# Caminho para a pasta contendo as imagens
pasta_imagens = os.path.join(diretorio_script, 'images')

# Lista todos os arquivos da pasta de imagens
try:
    imagens_arquivos = os.listdir(pasta_imagens)
except FileNotFoundError:
    raise FileNotFoundError(f"A pasta de imagens '{pasta_imagens}' não foi encontrada.")


# Preparação para exibir resultados
# melhorando a visualização
# ('Canny1', lambda img: cv2.Canny(img, 130, 130)),

filtros = [
    ('1', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)),
    ('2', lambda img: cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)),
    ('3', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=3)),
    ('4', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)),
    ('5', lambda img: cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)),
    ('6', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=5)),
    ('7', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=7)),
    ('8', lambda img: cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=7)),
    ('9', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=7)),
    ('10', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=9)),
    ('11', lambda img: cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=9)),
    ('12', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=9)),
    ('13', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=11)),
    ('14', lambda img: cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=11)),
    ('15', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=11)),
    ('16', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=13)),
    ('17', lambda img: cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=13)),
    ('18', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=13)),
    ('19', lambda img: cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=15)),
    ('20', lambda img: cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=15)),
]
num_filtros = len(filtros)

num_objetos_detectados = []

for filtro_nome, filtro_funcao in filtros:
    # Leitura da imagem escolhida
    img = cv2.imread(os.path.join(pasta_imagens, imagens_arquivos[0]))

    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {imagens_arquivos[0]}")
    
    # Convertendo para escala de cinza
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Aplicando o filtro 
    img_filtrada = filtro_funcao(img_gray.copy())

    # Limiarizaçãoq
    _, img_thres = cv2.threshold(img_filtrada, 128, 255, cv2.THRESH_BINARY)

    # Encontrando contornos
    countors, _ = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Contagem de contornos com área maior que 100
    count_up100 = 0
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 100:
            count_up100 += 1
            cv2.drawContours(img, cnt, -1, (255, 0, 0), 1)

    num_objetos_detectados.append(count_up100)

    # Visualização dos resultados
    cv2.imshow(f'Filtro: {filtro_nome}', img)
    cv2.imshow(f'Filtro: {filtro_nome} - Filtro', img_filtrada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Identificando o filtro com maior resultado
filtro_maior_resultado = max(num_objetos_detectados)
indice_maior_resultado = num_objetos_detectados.index(filtro_maior_resultado)
filtro_nome_maior_resultado = filtros[indice_maior_resultado][0]

# Criando um gráfico
plt.bar([filtro_nome for filtro_nome, _ in filtros], num_objetos_detectados, color=['red' if filtro == filtro_nome_maior_resultado else 'blue' for filtro in [filtro_nome for filtro_nome, _ in filtros]])
plt.text(indice_maior_resultado, filtro_maior_resultado, str(filtro_maior_resultado), ha='center', va='bottom', color='red')

plt.xlabel('Filtro')
plt.ylabel('Número de Objetos Detectados')
plt.title('Desempenho dos Filtros na Detecção de Objetos')
plt.show()