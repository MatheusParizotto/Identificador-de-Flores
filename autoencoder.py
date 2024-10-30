import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Caminhos para os dados de treino e validação
train_dir = r'C:/Users/Parizotto/OneDrive/Documentos/Classificador_de_plantas/data/train'
val_dir = r'C:/Users/Parizotto/OneDrive/Documentos/Classificador_de_plantas/data/validation'

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='input'  # Usamos 'input' porque no autoencoder a entrada é igual à saída
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='input'
)

# Autoencoder
input_shape = (224, 224, 3)
encoding_dim = 64  # Você pode ajustar conforme necessário

# Encoder
input_img = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = layers.Conv2D(encoding_dim, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Compilando o Autoencoder
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Treinamento
autoencoder.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20  # Ajuste conforme necessário
)

# Salvando o Autoencoder treinado
autoencoder.save('autoencoder_flores.h5')
print("Autoencoder treinado e salvo com sucesso!")
