# 방법 1 : torch version
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# er 33613 -- /home/funzin/study/test.py 
# True
# 1
# NVIDIA GeForce RTX 2080 Ti

# 방법 1-2 : torch version 2
from torch import cuda
assert cuda.is_available()
assert cuda.device_count() > 0
print(cuda.get_device_name(cuda.current_device()))

# 방법 2 : tensorflow version
import tensorflow as tf
tf.__version__

## 방법 2-1 : 모든 사용 가능한 GPU List 보기
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 방법 2-2
tf.config.list_physical_devices('GPU')

# 방법 2-3
tf.config.experimental.list_physical_devices('GPU')

# 방법 2-4
# tf.debugging.set_log_device_placement(True)
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
# print(c)

# 방법 3 : confirm Keras sees the GPU
# from keras import backend
# assert len(backend.tensorflow_backend._get_available_gpus()) > 0

# 주로 사용하는 코드 1
import tensorflow as tf
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
tf.config.list_physical_devices('GPU')

# 주로 사용하는 코드 2 : 인식한 GPU 개수 출력
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Pytorch
import torch

print(torch.cuda.device_count())
# 1

print(torch.cuda.get_device_name(0))
# GeForce RTX 2080 Ti

print(torch.cuda.is_available())
# True