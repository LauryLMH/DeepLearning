import tensorflow as tf
import numpy as np

# 初始化 5*5 数组
arr = np.array(np.arange(0,25))
arr = arr.reshape((5,5))
print(f"初始数组：\n{arr}")

# zero-padding
zero_padding = tf.pad(arr,[[1,1],[1,1]])
print(f"\n全零填充：\n{zero_padding}")

# reflect-padding
reflect_padding = tf.pad(arr,[[1,1],[1,1]],"REFLECT")
print(f"\n反射填充：\n{reflect_padding}")

# replicate-padding
replicate_padding = tf.pad(arr,[[1,1],[1,1]],"SYMMETRIC")
print(f"\n复制填充：\n{replicate_padding}")
