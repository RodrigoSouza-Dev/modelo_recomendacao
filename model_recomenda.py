# Instalação de dependências
!pip install -q -U kaggle
!mkdir ~/.kaggle
!echo '{"username":"<your_kaggle_username>","key":"<your_kaggle_api_key>"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d paramaggarwal/fashion-product-images-small
!unzip -q fashion-product-images-small.zip

# Pré-processamento dos dados
import os
import pandas as pd
from shutil import move

df = pd.read_csv('/content/styles.csv', usecols=['id','masterCategory']).astype(str)
os.makedirs('/content/Fashion_data/categories', exist_ok=True)

# Organiza imagens por categoria
for index, row in df.iterrows():
    image_id = row['id']
    category = row['masterCategory']
    src_path = f'/content/images/{image_id}.jpg'
    dst_path = f'/content/Fashion_data/categories/{category}/{image_id}.jpg'
    if os.path.exists(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        move(src_path, dst_path)

# Treinamento do modelo
import tensorflow as tf
import tensorflow_hub as hub

data_dir = '/content/Fashion_data/categories'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
N_FEATURES = 256

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
train_generator = datagen.flow_from_directory(data_dir, subset="training", shuffle=True, **dataflow_kwargs)
valid_generator = datagen.flow_from_directory(data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

MODULE_HANDLE = 'https://tfhub.dev/google/bit/m-r50x3/1'
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(N_FEATURES, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=valid_generator, epochs=5)

# Salvar o modelo
model.save('/content/bit_model')


