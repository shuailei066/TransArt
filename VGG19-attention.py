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
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D, Multiply, Conv2D, Reshape, Bidirectional, LSTM
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from sklearn.metrics import roc_auc_score

import shap

# 设置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

# 定义CBAM模块
def cbam_block(input_feature, ratio=8):
    channel_axis = -1
    channels = input_feature.shape[channel_axis]
    
    # 通道注意力模块
    shared_layer_one = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)(GlobalAveragePooling2D()(input_feature))
    shared_layer_two = Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(shared_layer_one)
    
    # 重新调整通道
    channel_attention = Multiply()([input_feature, Reshape((1, 1, channels))(shared_layer_two)])
    
    # 空间注意力模块
    avg_pool = tf.reduce_mean(channel_attention, axis=channel_axis, keepdims=True)
    max_pool = tf.reduce_max(channel_attention, axis=channel_axis, keepdims=True)
    concat = layers.Concatenate(axis=channel_axis)([avg_pool, max_pool])
    
    # 3x3卷积层
    spatial_attention = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(concat)
    
    # 重新调整空间
    cbam_output = Multiply()([channel_attention, spatial_attention])
    return cbam_output

# 创建ImageDataGenerator
train_datagenerator = ImageDataGenerator(rescale=1./255, validation_split=0.25, 
                                         zoom_range=0.2, horizontal_flip=True, 
                                         vertical_flip=True, rotation_range=30, 
                                         width_shift_range=0.1, height_shift_range=0.1)
valid_datagenerator = ImageDataGenerator(rescale=1./255)

# 加载数据
train_augmented = train_datagenerator.flow_from_directory(
    directory='/public/home/shenyin_wsb_2606/Art_class/train', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical', 
    subset='training', 
    shuffle=True, 
    seed=123, 
    color_mode='rgb', 
    save_to_dir='/public/home/shenyin_wsb_2606/Art_class/train_augmented', 
    save_format='png', 
    save_prefix='aug'
)

test_augmented = valid_datagenerator.flow_from_directory(
    directory='/public/home/shenyin_wsb_2606/Art_class/test', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical', 
    save_to_dir='/public/home/shenyin_wsb_2606/Art_class/test_augmented', 
    save_format='png', 
    save_prefix='aug'
)

# 加载预训练的VGG19模型（不包括顶层）
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结预训练模型的所有层
for layer in vgg19_base.layers:
    layer.trainable = False

# 创建一个输入层
input_layer = Input(shape=(224, 224, 3))

# 获取VGG19的输出
vgg19_output = vgg19_base(input_layer)

# 将CBAM模块应用于VGG19输出
cbam_output = cbam_block(vgg19_output)

# 将输出调整形状为2D以适应LSTM层
cbam_reshape = Reshape((cbam_output.shape[1] * cbam_output.shape[2], cbam_output.shape[3]))(cbam_output)

# 使用双向LSTM层处理CBAM输出
bilstm_output = Bidirectional(LSTM(128, return_sequences=False))(cbam_reshape)

# 添加自定义的全连接层
x = Dense(256, activation='relu')(bilstm_output)
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
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='/public/home/shenyin_wsb_2606/Art_class/best_vgg19_with_cbam.h5', monitor='val_loss', save_best_only=True)

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

image_path = '/public/home/shenyin_wsb_2606/Art_class/AiArtData/AiArtData/1.png'
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  # VGG19 的预处理

predictions = custom_model.predict(img_array)
predicted_class = np.argmax(predictions)
class_labels = list(train_augmented.class_indices.keys())

print("Predicted class:", class_labels[predicted_class])

# 绘制训练曲线
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('VGG19-CBAM-Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VGG19-CBAM-Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 3, 3)
plt.plot(history.history['val_auc'])
plt.title('VGG19-CBAM-Validation AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig('/public/home/shenyin_wsb_2606/Art_class/vgg19_cbam.png')  # VGG19 CBAM模型的文件名
plt.show()