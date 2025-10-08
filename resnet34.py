# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 23:47:51 2024

@author: 123
"""

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
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Add, Activation, Multiply, Lambda, Conv2D
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score

import shap

# 设置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 定义注意力机制
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)
    
    return Multiply()([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

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

# 加载预训练的VGG19模型（不包括顶层）
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结预训练模型的所有层
for layer in vgg19_base.layers:
    layer.trainable = False

# 创建一个输入层
input_layer = Input(shape=(224, 224, 3))

# 获取VGG19的输出
vgg19_output = vgg19_base(input_layer)

# 应用CBAM注意力模块
vgg19_attention = cbam_block(vgg19_output)

# 将输出展开
vgg19_flatten = Flatten()(vgg19_attention)

# 添加自定义的全连接层
x = Dense(256, activation='relu')(vgg19_flatten)
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
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/public/home/shenyin_wsb_2606/Art_class/best_attention.h5', monitor='val_loss', save_best_only=True)

# 准备验证数据用于AUC计算
validation_steps = test_augmented.n // test_augmented.batch_size
x_val, y_val = next(test_augmented)
for i in range(validation_steps - 1):
    x, y = next(test_augmented)
    x_val = np.concatenate((x_val, x))
    y_val = np.concatenate((y_val, y))

auc_callback = AUCCallback((x_val, y_val))

# 训练模型
history = custom_model.fit(
    train_augmented, 
    validation_data=test_augmented, 
    epochs=30, 
    verbose=1, 
    callbacks=[es, model_checkpoint, auc_callback]
)

# 预测单张图像
from tensorflow.keras.preprocessing import image

image_path = '/public/home/shenyin_wsb_2606/Art_class/AiArtData/AiArtData/zoomout_1.jpg'
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

predictions = custom_model.predict(img_array)
predicted_class = np.argmax(predictions)
class_labels = list(train_augmented.class_indices.keys())

print("Predicted class:", class_labels[predicted_class])

# 绘制训练曲线
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 3, 3)
plt.plot(history.history['val_auc'])
plt.title('Validation AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig('/public/home/shenyin_wsb_2606/Art_class/training_curves_with_attention_and_auc.png')
plt.show()

import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf

# 选择每个类别的1个样本
num_classes = len(class_labels)
samples_per_class = 10
shap_samples = []
sample_indices = []

for class_index in range(num_classes):
    class_samples = x_val[y_val.argmax(axis=1) == class_index]
    if len(class_samples) > 0:
        sample_index = np.random.choice(len(class_samples))
        shap_samples.append(class_samples[sample_index])
        sample_indices.append(sample_index)

shap_samples = np.array(shap_samples)

print("Shape of shap_samples:", shap_samples.shape)

# 创建背景数据集
background = x_val[:100]  # 使用100个样本作为背景

# 创建一个解释器
explainer = shap.GradientExplainer(custom_model, background)

# 计算SHAP值
shap_values = explainer.shap_values(shap_samples)


# 为每个类别的样本计算和可视化SHAP值
for i, class_name in enumerate(class_labels):
    # 创建一个新的图形
    plt.figure(figsize=(15, 10))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(shap_samples[i])
    plt.title(f"Original Image - Class: {class_name}")
    plt.axis('off')
    
    # 显示SHAP值红蓝图
    plt.subplot(1, 2, 2)
    
    # 确保shap_values[i]和shap_samples[i]是numpy数组
    shap_value = np.array(shap_values[i])
    shap_sample = np.array(shap_samples[i])
    
    # 如果shap_value是3D，需要增加一个维度
    if shap_value.ndim == 3:
        shap_value = np.expand_dims(shap_value, axis=0)
    if shap_sample.ndim == 3:
        shap_sample = np.expand_dims(shap_sample, axis=0)
    
    shap.image_plot(shap_value, shap_sample, show=False)
    plt.title(f"SHAP Values - Class: {class_name}")
    
    plt.tight_layout()
    plt.savefig(f'/public/home/shenyin_wsb_2606/Art_class/shap_values_plot_{class_name}.png')
    plt.close()

    # 保存SHAP值到文件
    np.save(f'/public/home/shenyin_wsb_2606/Art_class/shap_values_{class_name}.npy', shap_value)

print("SHAP analysis completed. Images and SHAP values have been saved.")



