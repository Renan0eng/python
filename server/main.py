from flask import Flask, request, jsonify
from PIL import Image
import cv2
import numpy as np
import os
import io
import uuid  # Importa a biblioteca uuid para gerar nomes únicos para as imagens

app = Flask(__name__)

def preProcessing(img):
    imgPre0 = cv2.Canny(img, 130, 130)
    return imgPre0

def reconhecer_objetos(img):
    img = cv2.resize(img, (640, 480))
    imgThres = preProcessing(img)

    countors, h1 = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count_up2000 = 0
    roi_images = []

    for idx, cnt in enumerate(countors):
        area = cv2.contourArea(cnt)
        if area > 100:
            count_up2000 += 1
            # Contorno verde dos objetos detectados
            cv2.drawContours(imgThres, cnt, -1, (0, 255, 0), 1)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            # Região de interesse (ROI) dos objetos detectados
            imgROI = img[y:y + h, x:x + w]
            roi_images.append(imgROI)

            # Sauva a ROI em um arquivo sendo o nome as coordenadas do retângulo e todos os dados para indentificalo na imagem
            cv2.imwrite(f"roi/{x}_{y}_{w}_{h}_{idx}.png", imgROI)


    result_image = img.copy()
    cv2.putText(result_image, "Objects: " + str(count_up2000), (10, 470), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 40, 0), 2)

    return result_image, roi_images, imgThres

@app.route('/processar_imagem', methods=['POST'])
def processar_imagem():
    try:
        # Verifica se a requisição contém um arquivo de imagem
        if 'imagem' not in request.files:
            return jsonify({'error': 'Nenhuma imagem encontrada'}), 400

        # Lê a imagem enviada na requisição
        imagem_enviada = request.files['imagem'].read()
        imagem_original = Image.open(io.BytesIO(imagem_enviada))
        imagem_array_original = np.array(imagem_original)

        # Gera um nome único para a imagem antes do processamento
        nome_unico_antes = str(uuid.uuid4())
        caminho_antes = f"imagens_antes/{nome_unico_antes}.png"

        # Salva a imagem original antes do processamento
        Image.fromarray(imagem_array_original).save(caminho_antes)

        # Chama a função de reconhecimento de objetos
        imagem_processada, roi_images, imgThres = reconhecer_objetos(imagem_array_original)

        # Gera um nome único para a imagem depois do processamento
        nome_unico_depois = str(uuid.uuid4())
        caminho_depois = f"imagens_depois/{nome_unico_depois}.png"

        # Salva a imagem com o filtro de Canny aplicado
        Image.fromarray(imgThres).save(f"imagens_depois/{nome_unico_depois}_canny.png")

        # Salva a imagem processada
        Image.fromarray(imagem_processada).save(caminho_depois)

        # Converte de RGBA para RGB, se necessário
        if imagem_processada.shape[2] == 4:
            imagem_processada = cv2.cvtColor(imagem_processada, cv2.COLOR_RGBA2RGB)

        # Salva a imagem processada temporariamente
        temp_output = io.BytesIO()
        Image.fromarray(imagem_processada).save(temp_output, format='JPEG')
        temp_output.seek(0)

        # Retorna a imagem processada e os caminhos das imagens antes e depois como resposta
        return jsonify({
            'imagem_processada': temp_output.read().decode('latin1'),
            'roi_images_count': len(roi_images),
            'imagem_antes': caminho_antes,
            'imagem_depois': caminho_depois
        })

    except Exception as e:
        # Trata exceções e retorna uma resposta de erro
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Cria pastas para armazenar imagens antes e depois do processamento
    os.makedirs("imagens_antes", exist_ok=True)
    os.makedirs("imagens_depois", exist_ok=True)

    # Inicia o servidor Flask em modo de depuração
    app.run(debug=True)
