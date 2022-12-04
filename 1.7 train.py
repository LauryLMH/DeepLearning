# 数据读取
from tensorflow.keras.datasets import mnist
(img_train, label_train), (img_test, label_test) = mnist.load_data()

# 数据可视化
import matplotlib.pyplot as plt
for i in range(5):
    plt.figure()
    plt.imshow(img_test[i], cmap='gray')
    plt.show()

# 模型搭建
import tensorflow as tf
# 模型建立
model = tf.keras.models.Sequential()
# 构建展平输入层
model.add(tf.keras.layers.Flatten())
# 构建全连接层：128个神经元，激活函数为relu
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# model.add(tf.keras.layers.Dense(units=256, activation='relu'))
# model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# 构建全连接层：10个神经元，激活函数为softmax
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# 模型编译
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# 模型训练
model.fit(img_train, label_train, batch_size=16, epochs=100,
          validation_data=(img_test,label_test), validation_freq=1)

# 模型预测
import numpy as np
predict_index = 0
print(f'标签：{label_test[predict_index]}')
result = model.predict(img_test[predict_index].reshape((1,28,28,1)))
result_index = np.argmax(result)
print(f'预测：{result_index}')