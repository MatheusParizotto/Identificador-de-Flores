from flask import Flask, request, render_template # type: ignore
import tensorflow as tf
import sqlite3
from PIL import Image
import numpy as np
import os
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)

# Carregar o modelo autoencoder 
try:
    autoencoder = tf.keras.models.load_model(r'C:/Users/Parizotto/OneDrive/Documentos/Classificador_de_plantas/autoencoder_flores.h5', custom_objects={'mse': MeanSquaredError()})
except Exception as e:
    print(f"Erro ao carregar o autoencoder: {e}")

# Carregar o modelo 
try:
    model = tf.keras.models.load_model(r'C:/Users/Parizotto/OneDrive/Documentos/Classificador_de_plantas/classificador_de_flores.h5')
except Exception as e:
    print(f"Erro ao carregar o modelo de classificação: {e}")

# Caminho para o upload de imagens
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Pré-processamento da imagem
            image = Image.open(image_path)
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # Mapeamento de classes
            class_mapping = {0: 'Dente-de-leão', 1: 'Girassol', 2: 'Margarida', 3: 'Rosa', 4: 'Tulipa'}

            reconstructed_image = autoencoder.predict(image)
            loss = np.mean(np.abs(image - reconstructed_image))

            limiar_anomalia = 0.02

            if loss > limiar_anomalia:
                nome_planta = "Desconhecida"
                cuidados_necessarios = "Informações não disponíveis."
            else:
                # Predição
                predictions = model.predict(image)
                confidence = np.max(predictions[0])
                classe_planta = np.argmax(predictions[0])

                limiar_confiança = 0.9

                if confidence < limiar_confiança:
                    nome_planta = "Desconhecida"
                    cuidados_necessarios = "Informações não disponíveis."
                else:
                    nome_planta = class_mapping[classe_planta]
                    # Conectar ao banco de dados
                    try:
                        conn = sqlite3.connect('data/cuidados.db')
                        cursor = conn.cursor()
                        cursor.execute("SELECT cuidados_necessarios FROM instrucoes WHERE nome_planta = ?", (nome_planta,))
                        resultado = cursor.fetchone()
                        if resultado:
                            cuidados_necessarios = resultado[0]
                        else:
                            cuidados_necessarios = "Informações não disponíveis."
                    except sqlite3.Error as e:
                        print(f"Erro ao conectar ao banco de dados: {e}")
                        nome_planta = "Erro"
                        cuidados_necessarios = "Não foi possível acessar o banco de dados."

            return render_template('index.html', image_path=image_path, nome_planta=nome_planta, cuidados_necessarios=cuidados_necessarios)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
