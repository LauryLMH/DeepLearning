# 导入压缩文件读取库
import gzip
# 读取测试集压缩文件
f = gzip.open('./data/t10k-images-idx3-ubyte.gz','r')
# 跳过前16个字节，因为是一些其他信息
print(f.read(16))
# 读取前五张图片的字节数据
num_img = 5
buf = f.read(28 * 28 * num_img)
# 导入numpy，将字节首先转换为uint8，再转换为float32类型
import numpy as np
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
# 调整数组形状
data = data.reshape(num_img, 28, 28, 1)

# 前5张测试图片可视化
import matplotlib.pyplot as plt
for i in range(num_img):
    plt.figure()
    plt.imshow(data[i], cmap='gray')
    plt.show()