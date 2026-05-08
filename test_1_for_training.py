import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#이미지 224*224
IMG_SIZE = 224
BATCH_SIZE = 32
total_run_times = 10
types_of_tanks = 10 #일단 탱크 10종으로 표시함

#이건 단순히 더 많은 데이터를 학습하도록 기존이미지를 살짝 변형하는거-사진을 살짝 돌리고 확대하고 ㅇ좌우 회전 시키는거임
tank_train_module = ImageDataGenerator(
    preprocessing_input=preprocess_input,#지금 맥 버츄얼 구성해도 잘 안되서 모르겠지마ㅏㄴ 자동 스케일러       
    rotation_range=10,   
    zoom_range=0.4,      
    horizontal_flip=True 
)


data_rescal = ImageDataGenerator(preprocessing_input=preprocess_input)

#데이터 읽어오기 디렉토리 변경은 나중에 하는걸로
tank_train_data = tank_train_module.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical" 
)

data_rescaled = data_rescal.flow_from_directory(
    "dataset/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

#요게 ResNet50 쓰는 이유->기본 트레이닝 된 걸 들고오는거(모양, 선, 색깔, 등등)
base_model = ResNet50(
    weights="imagenet",
    include_top=False, #이걸 나누는 라벨은 없애는 걸로
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


base_model.trainable = False#이건 구글링 해서 넣은건데 아직 잘 모르겠다

#학습 전 머리 달아주는 사전 작업
model = models.Sequential([
    base_model, #ResNet50
    layers.GlobalAveragePooling2D(),  #3차원->1차원으로 단순화 시키는 명령어?
    layers.Dense(256, activation='relu'), #뉴런(생각하는 장치?) 256개 생성
    layers.Dropout(0.5), #과부화 피하기 용 - 절반 쉬게하기              
    layers.Dense(types_of_tanks, activation='softmax') 
])

# 모델이 어떻게 생겼나 명세서 한 번 출력해 봅니다.
model.summary()


model.compile(
    optimizer='adam', #더 좋은 optimizer 있는지 찾는중
    loss='categorical_crossentropy', #이게 오차 구하는 공식-레딧
    metrics=['accuracy'] #밑에 학습 시 저장하는 기준을 정확도로 설정
)

#자동저장
checkpoint = callbacks.ModelCheckpoint(
    "tank_classifier_beta.h5", #파일 이름 설정
    save_best_only=True, #더 좋은게 나오면 계속 다시 저장       
    monitor='val_accuracy'   #정확도 수치 계속 검사  
)



#학습 모델
history = model.fit(
    tank_train_data,            
    validation_data=data_rescaled, 
    epochs=total_run_times,    #지금은 10번 
    callbacks=[checkpoint]      # 잘 학습 됬을 때만 저장
)

model.save("tank_classifier_final.h5")#마지막에 모델 저장


#최종 검증 (test set)
test_data = data_rescal.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

loss, accuracy = model.evaluate(test_data)
print(f"Test accuracy: {accuracy:.2%}")
