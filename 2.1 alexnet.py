import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 读取数据集
from tensorflow.keras.datasets import cifar10
(img_train, label_train), (img_test, label_test) = cifar10.load_data()
# 归一化数据集
img_train = img_train / 255.0
img_test = img_test / 255.0
# 数据预处理：从32*32*3缩放到227*227*3
# img_train_new = np.zeros((50000, 227,227,3),dtype='uint8')
# for i, img in tqdm(enumerate(img_train)):
#     img_train_new[i] = tf.image.resize(img, (227,227))
#
# img_test_new = np.zeros((10000, 227,227,3),dtype='uint8')
# for i, img in tqdm(enumerate(img_test)):
#     img_test_new[i] = tf.image.resize(img, (227,227))

# 配置标签
all_labels_name = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '卡车']

# 加载模型
# AlexNet_model = tf.keras.models.load_model('alexnet_model.h5')

# 搭建模型
AlexNet_model = tf.keras.Sequential([
    layers.Conv2D(96, kernel_size=(3, 3), input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    layers.Conv2D(384, kernel_size=(3, 3), strides=1,
                  padding='same', activation='relu'),

    layers.Conv2D(384, kernel_size=(3, 3), strides=1,
                  padding='same', activation='relu'),

    layers.Conv2D(256, kernel_size=(3, 3), strides=1,
                  padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

AlexNet_model.summary()

# 模型编译
AlexNet_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])

# 模型训练
AlexNet_model.fit(img_train, label_train, batch_size=16, epochs=30,
                 validation_data=(img_test, label_test), validation_freq=1)

AlexNet_model.save('alexnet_model.h5')

# 模型评估
# loss, accuracy = LeNet5_model.evaluate(img_train, label_train)
# print('\ntrain loss',loss)
# print('train accuracy',accuracy)
# loss, accuracy = LeNet5_model.evaluate(img_test, label_test)
# print('\ntest loss',loss)
# print('test accuracy',accuracy)

# 模型预测
import numpy as np

predict_index = 0
print(f'标签：{all_labels_name[label_test[predict_index][0]]}')
result = AlexNet_model.predict(img_test[predict_index].reshape((1, 32, 32, 3)))
result_index = np.argmax(result)
print(f'预测：{all_labels_name[result_index]}')
