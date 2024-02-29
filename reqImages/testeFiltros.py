import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

diretorio_script = os.path.dirname(__file__)
pasta_imagens = os.path.join(diretorio_script, 'images')

try:
    imagens_arquivos = os.listdir(pasta_imagens)
except FileNotFoundError:
    raise FileNotFoundError(f"A pasta de imagens '{pasta_imagens}' não foi encontrada.")

# Lista de filtros
filtros = [
    [
        ('cvtColor', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ('Canny', lambda img: cv2.Canny(img, 130, 130)),
    ],
    [
        ('cvtColor', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ('Canny', lambda img: cv2.Canny(img, 130, 130)),
        ('limiar', lambda img: cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]),
    ],
    [
        ('cvtColor', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ('Canny', lambda img: cv2.Canny(img, 130, 130)),
        ('limiar', lambda img: cv2.threshold(img, 100, 100, cv2.THRESH_BINARY)[1]),
    ],
    [
        ('cvtColor', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ('Canny', lambda img: cv2.Canny(img, 130, 130)),
        ('limiar', lambda img: cv2.threshold(img, 50, 200, cv2.THRESH_BINARY)[1]),
    ],
    [
        ('cvtColor', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ('Canny', lambda img: cv2.Canny(img, 130, 130)),
        ('limiar', lambda img: cv2.threshold(img, 200, 50, cv2.THRESH_BINARY)[1]),
    ],
]
num_filtros = len(filtros)

dados_combinacoes = []

for i, filtros_aplicar in enumerate(filtros):
    print(f'Filtros {i+1}: {filtros_aplicar}')
    # Leitura da imagem escolhida
    img = cv2.imread(os.path.join(pasta_imagens, imagens_arquivos[0]))

    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {imagens_arquivos[0]}")

    # Medindo o tempo inicial
    start_time = time.time()

    # Aplicando os filtros percorrendo a lista de filtros
    for filtro_nome, filtro in filtros_aplicar:
        print(f'Aplicando filtro {filtro_nome}')
        img = filtro(img)

    # Encontrando contornos
    countors, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Contagem de contornos com área maior que 100
    count_up100 = 0
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 100:
            count_up100 += 1
            cv2.drawContours(img, cnt, -1, (255, 0, 0), 1)

    # Medindo o tempo final
    end_time = time.time()
    tempo_processamento = end_time - start_time

    dados_combinacoes.append({
        'filtro': filtros_aplicar,
        'num_objetos_detectados': count_up100,
        'tempo_processamento': tempo_processamento,
    })

    # Visualização dos resultados
    cv2.imshow(f'Filtros {i+1}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Identificando a combinação de filtros com maior resultado
max_combinacao = max(dados_combinacoes, key=lambda x: x['num_objetos_detectados'])
indice_maior_resultado = dados_combinacoes.index(max_combinacao)

# Exibindo informações sobre a combinação com maior resultado
print(f"Combinação com mais resultados:")
print(f" - Filtros aplicados: {max_combinacao['filtro']}")
print(f" - Número de objetos detectados: {max_combinacao['num_objetos_detectados']}")
print(f" - Tempo de processamento: {max_combinacao['tempo_processamento']} segundos")

# Criando um gráfico
fig, ax = plt.subplots()
bars = ax.bar([f'Filtros {i+1}' for i in range(num_filtros)], [d['num_objetos_detectados'] for d in dados_combinacoes], color=['red' if i == indice_maior_resultado else 'blue' for i in range(num_filtros)])

for bar, data in zip(bars, dados_combinacoes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f"{data['num_objetos_detectados']}\n{data['tempo_processamento']:.2f}s", ha='center', va='bottom', color='black')

plt.xlabel('Combinação de Filtros')
plt.ylabel('Número de Objetos Detectados')
plt.title('Desempenho das Combinações de Filtros na Detecção de Objetos')
plt.show()
