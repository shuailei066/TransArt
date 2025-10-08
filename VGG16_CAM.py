import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score

import shap

# 设置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建一个自定义回调来计算验证集的AUC
class AUCCallback(Callback):
    def __init__(self, validation_data):
        super(AUCCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        auc = roc_auc_score(y_val, y_pred, average='weighted', multi_class='ovr')
        print(f'\nValidation AUC: {auc:.4f}')
        logs['val_auc'] = auc

# 创建ImageDataGenerator
train_datagenerator = ImageDataGenerator(rescale=1./255, validation_split=0.25, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1)
valid_datagenerator = ImageDataGenerator(rescale=1./255)

# 加载数据
train_augmented = train_datagenerator.flow_from_directory(directory='/public/home/shenyin_wsb_2606/Art_class/train', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training', shuffle=True, seed=123, color_mode='rgb', save_to_dir='/public/home/shenyin_wsb_2606/Art_class/train_augmented', save_format='png', save_prefix='aug')
test_augmented = valid_datagenerator.flow_from_directory(directory='/public/home/shenyin_wsb_2606/Art_class/test', target_size=(224, 224), batch_size=32, class_mode='categorical', save_to_dir='/public/home/shenyin_wsb_2606/Art_class/test_augmented', save_format='png', save_prefix='aug')

# 加载预训练的VGG16模型（不包括顶层）
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结预训练模型的所有层
for layer in vgg16_base.layers:
    layer.trainable = False

# 创建一个输入层
input_layer = Input(shape=(224, 224, 3))

# 获取VGG16的输出
vgg16_output = vgg16_base(input_layer)

# 将输出展开
vgg16_flatten = Flatten()(vgg16_output)

# 添加自定义的全连接层
x = Dense(256, activation='relu')(vgg16_flatten)
x = Dense(128, activation='relu')(x)
output_layer = Dense(train_augmented.num_classes, activation='softmax')(x)

# 创建最终模型
custom_model = Model(inputs=input_layer, outputs=output_layer)

# 打印模型结构
custom_model.summary()

# 编译模型
custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

# 设置回调
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/public/home/shenyin_wsb_2606/Art_class/best_vgg16.h5', monitor='val_loss', save_best_only=True)

# 准备验证数据用于AUC计算
validation_steps = test_augmented.n // test_augmented.batch_size
x_val, y_val = next(test_augmented)
for i in range(validation_steps - 1):
    x, y = next(test_augmented)
    x_val = np.concatenate((x_val, x))
    y_val = np.concatenate((y_val, y))

auc_callback = AUCCallback((x_val, y_val))

# # 训练模型
# history = custom_model.fit(
#     train_augmented, 
#     validation_data=test_augmented, 
#     epochs=30, 
#     verbose=1, 
#     callbacks=[es, model_checkpoint, auc_callback]
# )

# 预测单张图像
from tensorflow.keras.preprocessing import image

image_path = '/public/home/shenyin_wsb_2606/Art_class/AiArtData/AiArtData/zoomout_1.jpg'
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

predictions = custom_model.predict(img_array)
predicted_class = np.argmax(predictions)
class_labels = list(train_augmented.class_indices.keys())

print("Predicted class:", class_labels[predicted_class])


def model_predict(images):
    images = tf.keras.applications.vgg16.preprocess_input(images)
    return custom_model.predict(images)

import shap

# 加载并预处理图像
img_path = '/public/home/shenyin_wsb_2606/Art_class/AiArtData/AiArtData/zoomout_1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# 创建背景数据集
background = np.zeros((20, 224, 224, 3))  # 20 是样本数量，可以根据需要调整

# 初始化 SHAP explainer
explainer = shap.DeepExplainer(custom_model, background)

# 计算 SHAP 值
shap_values = explainer.shap_values(img_array)

# 获取预测结果
predictions = custom_model.predict(img_array)
predicted_class = np.argmax(predictions[0])
class_labels = list(train_augmented.class_indices.keys())

# 绘制 SHAP 值
plt.figure(figsize=(16, 8))

plt.subplot(121)
shap.image_plot(shap_values, -img_array, show=False)
plt.title(f"SHAP values for class: {class_labels[predicted_class]}")

plt.subplot(122)
plt.imshow(img_array[0] / 255)  # 显示原始图像
plt.title("Original Image")
plt.axis('off')

plt.tight_layout()
plt.savefig('/public/home/shenyin_wsb_2606/Art_class/shap_explanation.png')
plt.show()

print(f"Predicted class: {class_labels[predicted_class]}")
print(f"Confidence: {predictions[0][predicted_class]:.2f}")