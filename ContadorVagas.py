from itertools import count
from tabnanny import check

import cv2  # Biblioteca para processamento de imagens e vídeo
import pickle  # Para salvar e carregar objetos Python, como listas
import numpy as np  # Biblioteca para cálculos matemáticos e manipulação de arrays

from cv2.gapi import kernel  # (não utilizado no código, pode ser removido)

# Lista para armazenar as regiões de interesse (vagas de estacionamento)
vagas = []

# Carrega as coordenadas das vagas de um arquivo chamado 'vagas.pkl'
with open('vagas.pkl', 'rb') as arquivo:
    vagas = pickle.load(arquivo)

# Carrega um vídeo chamado 'video.mp4'
video = cv2.VideoCapture('video.mp4')

# Loop principal para processar cada frame do vídeo
while True:
    # Lê um frame do vídeo
    check, img = video.read()

    # Converte o frame para escala de cinza (mais fácil para processamento de imagem)
    imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplica um limiar adaptativo para criar uma imagem binária (preto e branco)
    imgTh = cv2.adaptiveThreshold(imgCinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 16)

    # Remove ruídos usando um filtro mediano
    imgMedian = cv2.medianBlur(imgTh, 5)

    # Cria um kernel (matriz) para dilatação
    kernel = np.ones((3, 3), np.int8)

    # Aplica dilatação para aumentar as áreas brancas na imagem binária
    imgDil = cv2.dilate(imgMedian, kernel)

    # Contador para vagas disponíveis
    vagasAbertas = 0

    # Percorre todas as regiões definidas como vagas no arquivo 'vagas.pkl'
    for x, y, w, h in vagas:
        # Extrai a região correspondente à vaga da imagem dilatada
        vaga = imgDil[y:y + h, x:x + w]

        # Conta os pixels brancos na região (área ocupada)
        count = cv2.countNonZero(vaga)

        # Exibe o número de pixels brancos na imagem original (como texto)
        cv2.putText(img, str(count), (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        # Define se a vaga está livre (poucos pixels brancos) ou ocupada
        if count < 900:  # Limite para considerar a vaga como "livre"
            # Desenha um retângulo verde ao redor da vaga livre
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vagasAbertas += 1  # Incrementa o contador de vagas disponíveis
        else:
            # Desenha um retângulo vermelho ao redor da vaga ocupada
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Adiciona uma barra de informações no canto superior esquerdo
    cv2.rectangle(img, (90, 0), (415, 60), (0, 255, 0), -1)
    cv2.putText(img, f'LIVRE: {vagasAbertas}/69', (95, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

    # Mostra o frame original com as anotações
    cv2.imshow('video', img)

    # Mostra o frame processado (imagem binária dilatada)
    cv2.imshow('video Th', imgDil)

    # Aguarda 10ms para permitir a exibição (e parar com uma tecla, se necessário)
    cv2.waitKey(10)