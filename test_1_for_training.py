import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ===== 설정값 =====
IMG_SIZE = 224
BATCH_SIZE = 32           # 데이터 1000장 미만이면 줄이기
EPOCHS_HEAD = 10          
EPOCHS_FINETUNE = 15      
UNFREEZE_LAYERS = 30      
AUTOTUNE = tf.data.AUTOTUNE


# ===== 데이터 읽어오기 =====
train_ds = keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# 클래스 순서는 prefetch 붙이기 전에 뽑아둬야 함
class_names = train_ds.class_names
print("클래스 순서:", class_names)
types_of_tanks = len(class_names)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ===== 증강 =====
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="augment")


# ===== 베이스 모델 =====
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False   # 1단계에서는 통째로 얼려둠


# ===== 모델 조립 =====
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)                    # ResNet50 전용 스케일러
x = base_model(x, training=False)          # BatchNorm 추론 모드 고정
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(types_of_tanks, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()


# ===== 콜백 =====
cb_list = [
    keras.callbacks.ModelCheckpoint(
        "tank_classifier_best.keras",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]


# ===== 1단계: 머리만 학습 =====
print("\n===== 1단계: 헤드 학습 =====")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=cb_list
)


# ===== 2단계: 뒷블록 풀고 미세조정 =====
print("\n===== 2단계: 미세조정 =====")
base_model.trainable = True

for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=cb_list
)

model.save("tank_classifier_final.keras")


# ===== 최종 검증 =====
loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy:.2%}")


# ===== 학습 곡선 =====
acc = history_head.history["accuracy"] + history_ft.history["accuracy"]
val_acc = history_head.history["val_accuracy"] + history_ft.history["val_accuracy"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="train")
plt.plot(val_acc, label="val")
plt.axvline(len(history_head.history["accuracy"]) - 1,
            color="gray", linestyle="--", label="finetune start")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("training_curve.png")
plt.show()
