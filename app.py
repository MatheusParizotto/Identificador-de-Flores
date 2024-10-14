from flask import Flask, request, render_template
import tensorflow as tf
import sqlite3
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Carregar o modelo treinado
try:
    model = tf.keras.models.load_model(r'C:/Users/Parizotto/OneDrive/Documentos/Classificador_de_plantas/classificador_de_flores.h5')
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

id_procurado = 1

# Configura o caminho para o upload de imagens
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

            # Fazer a predição
            predictions = model.predict(image)
            classe_planta = np.argmax(predictions[0])
            
            print("Classe Predita (índice):", classe_planta)

            # Verificar a conexão com o banco de dados
            try:
                conn = sqlite3.connect('data/cuidados.db')
                cursor = conn.cursor()
                print("Conexão com o banco de dados estabelecida com sucesso!")
                
                cursor.execute("SELECT nome_planta, cuidados_necessarios FROM instrucoes WHERE id = ?", (id_procurado,))
                resultado = cursor.fetchone()
                
                if resultado:
                    nome_planta, cuidados_necessarios = resultado
                else:
                    nome_planta = "Desconhecida"
                    cuidados_necessarios = "Informações não disponíveis."
                    
            except sqlite3.Error as e:
                print(f"Erro ao conectar ao banco de dados: {e}")
                nome_planta = "Erro"
                cuidados_necessarios = "Não foi possível acessar o banco de dados."
                
            print("Resultado", resultado)
            print("Nome da Planta:", nome_planta)
            print("Cuidados Necessários:", cuidados_necessarios)

            return render_template('index.html', image_path=image_path, nome_planta=nome_planta, cuidados_necessarios=cuidados_necessarios)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)