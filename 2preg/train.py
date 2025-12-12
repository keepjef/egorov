import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================= КОНФИГУРАЦИЯ =================
DATASET_DIR = 'dataset_mirror'   # Корневая папка с данными
IMG_SIZE = 256            # Размер изображения для входа сети
BATCH_SIZE = 64            # Размер батча (уменьшите до 4 или 2, если мало памяти)
EPOCHS = 20               # Количество эпох обучения
LEARNING_RATE = 1e-3      # Скорость обучения

# Имена файлов, которые мы ищем в папках
INPUT_FILENAME = '1.png'
TARGET_FILENAME = '2.png'
# ================================================

def get_image_pairs(root_dir):
    """
    Проходит по всем подпапкам root_dir и ищет пары файлов 1.png и 2.png.
    Возвращает список кортежей: [(path_to_1, path_to_2), ...]
    """
    pairs = []
    print(f"Поиск данных в: {os.path.abspath(root_dir)} ...")
    
    for dirpath, _, filenames in os.walk(root_dir):
        if INPUT_FILENAME in filenames and TARGET_FILENAME in filenames:
            input_path = os.path.join(dirpath, INPUT_FILENAME)
            target_path = os.path.join(dirpath, TARGET_FILENAME)
            pairs.append((input_path, target_path))
            
    return pairs

def load_image(path):
    """Читает файл, декодирует PNG, меняет размер и нормализует."""
    img_bytes = tf.io.read_file(path)
    # channels=3 принудительно делает RGB, даже если картинка ч/б
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) # [0, 1]
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def process_pair(input_path, target_path):
    """Загружает пару путей и возвращает пару тензоров (x, y)."""
    input_img = load_image(input_path)
    target_img = load_image(target_path)
    return input_img, target_img

def create_dataset(pairs, batch_size, shuffle=False):
    """Создает tf.data.Dataset из списка путей."""
    input_paths, target_paths = zip(*pairs)
    
    ds = tf.data.Dataset.from_tensor_slices((list(input_paths), list(target_paths)))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs))
    
    # num_parallel_calls ускоряет загрузку данных
    ds = ds.map(process_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- Архитектура U-Net (как в исходном ноутбуке) ---
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPool2D(2)(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    # Output
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(d4)

    model = keras.Model(inputs, outputs, name='unet_mirror')
    return model

if __name__ == "__main__":
    # 1. Проверяем доступность GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # 2. Ищем файлы
    pairs = get_image_pairs(DATASET_DIR)
    
    if not pairs:
        print(f"ОШИБКА: Не найдено пар '{INPUT_FILENAME}' и '{TARGET_FILENAME}' в папке '{DATASET_DIR}' или подпапках.")
        exit(1)

    print(f"Найдено пар изображений: {len(pairs)}")
    
    # 3. Делим на Train/Val (90% / 10%)
    random.seed(42)
    random.shuffle(pairs)
    
    val_split = int(len(pairs) * 0.1)
    train_pairs = pairs[val_split:]
    val_pairs = pairs[:val_split]
    
    print(f"Тренировка: {len(train_pairs)}, Валидация: {len(val_pairs)}")

    # 4. Создаем датасеты
    train_ds = create_dataset(train_pairs, BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(val_pairs, BATCH_SIZE, shuffle=False)

    # 5. Строим и компилируем модель
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mae',  # Mean Absolute Error хорошо подходит для pixel-to-pixel задач
        metrics=['accuracy']
    )

    # 6. Запускаем обучение
    print("Начинаем обучение...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )
        # 7. Сохраняем модель
        save_path = 'unet_custom_dataset.h5'
        model.save(save_path)
        print(f"Модель успешно сохранена в: {save_path}")
        
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
        model.save('unet_interrupted.h5')
        print("Промежуточный результат сохранен в unet_interrupted.h5")