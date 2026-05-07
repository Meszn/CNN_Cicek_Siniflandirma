"""
flowers dataset: 
    - rgb: 224x224
    - 5 classes

CNN ile siniflandirma modeli olusturma ve problemi cozme 

"""
#import libraries
from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,# convolutional layer
    MaxPool2D, # pooling layer
    Flatten, # flattening layer
    Dense,# fully connected layer
    Dropout # dropout layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,# early stopping
    ReduceLROnPlateau,# ogrenme oranini azaltma
    ModelCheckpoint # model checkpoint
)

import tensorflow as tf
import matplotlib.pyplot as plt

#veriyi yukleme
(ds_train , ds_val), ds_info = load(
    'tf_flowers',
    split=["train[:80%]", "train[80%:]"],# veriyi %80 train, %20 test olarak bolme
    as_supervised=True,#veriyi (image, label) formatinda alma
    with_info=True #veri seti hakkinda bilgi alma
)
print(ds_info._features)
print( "Number of classes: ", ds_info.features['label'].num_classes)

#eğitim setinden random 3 resim ve etiket alma
fig = plt.figure(figsize=(10,5))
for i, (image, label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1, 3, i+1)
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title(f"Label: {label.numpy()}")
    ax.axis("off")

plt.tight_layout()
plt.show()

IMG_SIZE = (180, 180)

#data augmentation + preprocessing
def preprocess_train(image, label):
    """
    resize, random flip, brightness, contrast, crop, normalization
    """
    image = tf.image.resize(image, IMG_SIZE) # resize
    image = tf.image.random_flip_left_right(image) # YATAY olarak rastgele çevirme
    image = tf.image.random_brightness(image, max_delta=0.1) # parlaklığı rastgele değiştirme
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2) # kontrastı rastgele değiştirme
    image = tf.image.random_crop(image, size=IMG_SIZE + (3,)) # rastgele kırpma
    image = tf.cast(image, tf.float32) / 255.0 # normalizasyon
    return image, label

def preprocess_val(image, label):
    """
    resize, normalization
    """
    image = tf.image.resize(image, IMG_SIZE) # resize
    image = tf.cast(image, tf.float32) / 255.0 # normalizasyon
    return image, label

ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE) # veri ön işleme ve augmentation
    .shuffle(1000) # veriyi karıştırma
    .batch(32) # batch boyutu
    .prefetch(AUTOTUNE)
)

ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)

#modeli olusturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)), # convolutional layer
    MaxPool2D((2, 2)), # pooling layer
    Conv2D(64, (3, 3), activation='relu'), # convolutional layer
    MaxPool2D((2, 2)), # pooling layer
    Conv2D(128, (3, 3), activation='relu'), # convolutional layer
    MaxPool2D((2, 2)), # pooling layer
    Flatten(), # flattening layer cok boyutlu veriyi tek boyuta indirme
    Dense(128, activation='relu'), # fully connected layer
    Dropout(0.5), # dropout layer
    Dense(ds_info.features['label'].num_classes, activation='softmax') # output layer
])

# (3x3) filtre neden kullanilir?
# - 3x3 filtreler, görüntüdeki küçük detayları yakalamak
# - Daha büyük filtrelere göre daha az parametre içerirler, bu da aşırı öğrenmeyi azaltır
# 32, 64, 128 filtre sayıları neden kullanilir?
# - Katman derinleştikçe, daha fazla filtre kullanarak daha karmaşık özellikleri öğrenmek
# - İlk katmanlarda daha az filtre, sonraki katmanlarda daha fazla filtre kullanmak yaygın bir uygulamadır
# MaxPooling neden kullanilir?
# - Görüntünün boyutunu azaltarak hesaplama maliyetini düşürmek
# - Önemli özellikleri koruyarak gürültüyü azaltmak
#dropout neden kullanilir?
# - Aşırı öğrenmeyi önlemek için rastgele nöronları devre dışı bırakmak
# - Modelin genelleme yeteneğini artırmak
# genel mantık: hesap kapasite ve ifade arasında denge kurmak
# daha fazla filtre ve katman daha fazla ifade gücü sağlar, ancak aşırı öğrenmeye neden olabilir
# daha az filtre ve katman daha az ifade gücü sağlar, ancak aşırı öğrenme riski daha düşüktür.
# bu nedenle, model mimarisi tasarlarken, veri setinin büyüklüğü ve karmaşıklığına göre uygun bir denge sağlamak önemlidir.

#callback'ler
callbacks = [
    #eger val_loss 3 epoch boyunca iyilesmezse egitimi durdur ve en iyi model agirlariyla geri don
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), # erken durdurma
    #eger val_loss 2 epoch boyunca iyilesmezse ogrenme oranini yarıya indir
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-9), # ogrenme oranini azaltma
    #egitim sırasında en iyi model agirlariyla kaydet
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True) # model checkpoint
]

#modeli derleme
model.compile(
    optimizer=Adam(learning_rate=0.001), # optimizer
    loss='sparse_categorical_crossentropy', # loss function
    metrics=['accuracy'] # evaluation metric
)

print(model.summary())

#modeli egitme
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=10,
    callbacks=callbacks, 
    verbose=1
)
#modeli degerlendirme
plt.figure(figsize=(12, 5))

#dogruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_history_accuracy.png') # grafikleri kaydet

#kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history_loss.png') # grafikleri kaydet
plt.show()