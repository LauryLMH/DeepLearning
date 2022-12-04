import tensorflow as tf
from tensorflow.keras import layers

# 读取数据集
from tensorflow.keras.datasets import cifar10
(img_train, label_train), (img_test, label_test) = cifar10.load_data()
# 归一化数据集
img_train = img_train / 255.0
img_test = img_test / 255.0
# 配置标签
all_labels_name = ['飞机', '汽车', '鸟', '猫', '鹿','狗', '蛙', '马', '船', '卡车']

# 加载模型
# LeNet5_model = tf.keras.models.load_model('lenet_model.h5')

# 搭建模型
LeNet5_model = tf.keras.Sequential([
    layers.Conv2D(input_shape=((32, 32, 3)),
                 filters=6, kernel_size=(5,5), strides=(1,1), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding='same'),
    layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2),padding='same'),
    layers.Conv2D(filters=120, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'),
    layers.Flatten(),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

LeNet5_model.summary()

# 模型编译
LeNet5_model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 模型训练
LeNet5_model.fit(img_train, label_train, batch_size=16, epochs=30,
          validation_data=(img_test,label_test), validation_freq=1)
LeNet5_model.save('model.h5')

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
result = LeNet5_model.predict(img_test[predict_index].reshape((1,32,32,3)))
result_index = np.argmax(result)
print(f'预测：{all_labels_name[result_index]}')
