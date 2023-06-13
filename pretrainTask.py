"""
@Time    : 2023/5/31 9:56
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: pretrainTask.py
@Software: PyCharm
"""
import math
import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from models.vaemodel import VAE
from models.base import BaseClassification
from tensorflow import keras
from keras.saving.save import load_model
from options import Options
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class preTrainTask(BaseClassification):

    def __init__(self, opt,latent_dim,epochs):

        super(preTrainTask, self).__init__(opt)
        self.latent_dim = latent_dim
        self.featuresize = 91
        self.epochs = epochs
        self.batch_size = 32

    def compute_loss(self,model, x) -> tf.Tensor:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_logit))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        total_loss = reconstruction_loss + kl_loss
        return total_loss

    @tf.function
    def compute_loss(self,model, x)\
            -> tf.Tensor:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_logit))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        total_loss = reconstruction_loss + kl_loss

        return total_loss

    @tf.function
    def train_step(self,model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
            # print('loss:', loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    def train(self,train_dataset,test_dataset):

        optimizer = tf.keras.optimizers.Adam(0.000008)
        model = VAE(self.latent_dim, self.featuresize)
        model.encoder.summary()
        model.decoder.summary()
        for epoch in range(1, self.epochs + 1):
            # 训练模型
            start_time = time.time()
            for train_x in train_dataset:
                self.train_step(model, train_x, optimizer)
            end_time = time.time()

            # 评估模型
            loss = tf.keras.metrics.Mean()
            # print('loss:', loss)
            for test_x in test_dataset:
                loss(self.compute_loss(model, test_x))
                # print('loss:', loss)
            elbo = -loss.result()
            # display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch, elbo, end_time - start_time))
            # generate_and_save_images(model, epoch, test_sample)

        model.encoder.save('vae_encoder.h5')
        # model.decoder.save('vae_decoder.h5')

    def fit(self,path):
        raw_data = pd.read_csv(path)
        dataframe = raw_data.replace([np.inf, -np.inf], np.nan).dropna().copy()
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        labels = dataframe.pop('appname')
        # columns = dataframe.columns
        datanew = self._process_standard(dataframe)
        train, test = train_test_split(datanew, test_size=0.1, random_state=0)

        train = train.astype('float32').reshape(-1, self.featuresize)
        test = test.astype('float32').reshape(-1, self.featuresize)
        train_dataset = (tf.data.Dataset.from_tensor_slices(train)
                         .batch(self.batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test)
                        .batch(self.batch_size))
        task = preTrainTask(opt,latent_dim=18,epochs = 80)
        task.train(train_dataset, test_dataset)


# def processStandard(data):
#     scaler = StandardScaler()
#     return scaler.fit_transform(data)
#
# def z_score_normalize(featurelist,dataframe1)\
#         -> DataFrame:
#
#     data_std = dataframe1.std()
#     data_mean = dataframe1.mean()
#     for _ in featurelist:
#         if data_std[_] == 0:
#             dataframe1[_] = 0
#         else:
#             dataframe1[_] = dataframe1[_].map(lambda x: (x - data_mean[_]) / data_std[_])
#
#     return dataframe1
#
# def log_normal_pdf(sample, mean, logvar, raxis=1):
#     log2pi = tf.math.log(2. * np.pi)
#     return tf.reduce_sum(
#         -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#         axis=raxis)
#

#
# @tf.function
# def train_step(model, x, optimizer):
#     """Executes one training step and returns the loss.
#
#     This function computes the loss and gradients, and uses the latter to
#     update the model's parameters.
#     """
#     with tf.GradientTape() as tape:
#         loss = compute_loss(model, x)
#         # print('loss:', loss)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# def main():
#
#     # 加载数据集
#     raw_data = pd.read_csv('./csv_data/dataframe7.csv', sep=',', header='infer')
#     dataframe = raw_data.replace([np.inf, -np.inf], np.nan).dropna().copy()
#     data = dataframe.drop(dataframe.columns[[0, 92]], axis=1)
#     columns = data.columns
#     print('start processing normalize!')
#     datanew = processStandard(data)
#     print('finish processing!')
#     train, test = train_test_split(datanew, test_size=0.1, random_state=0)
#
#     featuresize = 91
#     train = train.astype('float32').reshape(-1, featuresize)
#     test = test.astype('float32').reshape(-1, featuresize)
#
#     # train_norm = z_score_normalize(columns,train)
#     # test_norm = z_score_normalize(columns,test)
#
#     # train_norm = min_max_normalization(train)
#     # test_norm = min_max_normalization(test)
#
#     # (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
#     # train_images = preprocess_images(train_images)
#     # test_images = preprocess_images(test_images)
#
#     # 训练参数
#     train_size = 60000
#     batch_size = 32
#     test_size = 10000
#
#     # 数据分批并打乱
#     train_dataset = (tf.data.Dataset.from_tensor_slices(train)
#                      .shuffle(train_size).batch(batch_size))
#     test_dataset = (tf.data.Dataset.from_tensor_slices(test)
#                     .shuffle(test_size).batch(batch_size))
#
#     optimizer = tf.keras.optimizers.Adam(0.000008)
#     epochs = 50
#     # set the dimensionality of the latent space to a plane for visualization later
#     latent_dim = 10
#     # num_examples_to_generate = 16
#
#     # # keeping the random vector constant for generation (prediction) so
#     # # it will be easier to see the improvement.
#     # random_vector_for_generation = tf.random.normal(
#     #     shape=[num_examples_to_generate, latent_dim])
#     model = VAE(latent_dim, featuresize)
#     model.encoder.summary()
#     model.decoder.summary()
#
#     # # Pick a sample of the test set for generating output images
#     # assert batch_size >= num_examples_to_generate
#     # for test_batch in test_dataset.take(1):
#     #     test_sample = test_batch[0:num_examples_to_generate, :]
#     #
#     # generate_and_save_images(model, 0, test_sample)
#
#     for epoch in range(1, epochs + 1):
#         # 训练模型
#         start_time = time.time()
#         for train_x in train_dataset:
#             train_step(model, train_x, optimizer)
#         end_time = time.time()
#
#         # 评估模型
#         loss = tf.keras.metrics.Mean()
#         # print('loss:', loss)
#         for test_x in test_dataset:
#             loss(compute_loss(model, test_x))
#             # print('loss:', loss)
#         elbo = -loss.result()
#         # display.clear_output(wait=False)
#         print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
#               .format(epoch, elbo, end_time - start_time))
#         # generate_and_save_images(model, epoch, test_sample)
#
#     model.encoder.save('vae_encoder.h5')
#     model.decoder.save('vae_decoder.h5')

class MyModel(keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.cov = keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu', input_shape=[6, 6, 1])
    self.pool = keras.layers.MaxPool2D(pool_size=(2,2))
    self.flatten = keras.layers.Flatten()
    self.d1 = keras.layers.Dense(128, activation='relu')
    self.drop = keras.layers.Dropout(0.05)
    self.d2 = keras.layers.Dense(7, activation='softmax')

  def call(self, x,training=None, mask=None):

    x = self.cov(x)
    x = self.drop(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.drop(x)
    x = self.d2(x)

    return x

class mlpTask(BaseClassification):

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.batch_size = 32

    def df_to_dataset(self,train, batch_size):
        y_train = train.pop('appname')
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        # y_train = keras.utils.to_categorical(y_train, 7)
        newx = train.values
        newx = newx.reshape((len(newx),6,6,1))
        ds = tf.data.Dataset.from_tensor_slices((newx, y_train))
        ds = ds.batch(batch_size)
        return ds


    def train_step(self,model,images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self,model,images, labels):
      # training=False is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self):
        raw_data = pd.read_csv('./csv_data/dataframe7.csv')
        dataframe = raw_data.replace([np.inf, -np.inf], np.nan).dropna().copy()
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        labels = dataframe.pop('appname')
        # columns = dataframe.columns
        datanew = self._process_standard(dataframe)
        test_model = load_model('./vae_encoder.h5')
        prediction = test_model.predict(datanew, verbose=1)

        encoderdataframe = DataFrame(prediction)
        encoderdataframe['appname'] = labels.values
        # print(prediction)
        train, test = train_test_split(encoderdataframe, test_size=0.2)

        # train
        ds = self.df_to_dataset(train,self.batch_size)
        # y_train = train.pop('appname')
        # le = LabelEncoder()
        # y_train = le.fit_transform(y_train)
        # # y_train = keras.utils.to_categorical(y_train, 7)
        # newx = train.values
        # # newx = newx.reshape((len(newx),1,20))
        # ds = tf.data.Dataset.from_tensor_slices((newx, y_train))
        # ds = ds.batch(self.batch_size)

        # test
        test_ds = self.df_to_dataset(test, self.batch_size)
        # SparseCategoricalCrossentropy
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model = MyModel()
        # optimizer = tf.keras.optimizers.Adam()
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # train_loss = tf.keras.metrics.Mean(name='train_loss')
        # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        #
        # test_loss = tf.keras.metrics.Mean(name='test_loss')
        # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        EPOCHS = 50

        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in ds:
                self.train_step(model,images, labels)

            for test_images, test_labels in test_ds:
                self.test_step(model,test_images, test_labels)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Accuracy: {self.train_accuracy.result() * 100}, '
                f'Test Loss: {self.test_loss.result()}, '
                f'Test Accuracy: {self.test_accuracy.result() * 100}'
            )

# def df_to_dataset(train,batch_size):
#
#     y_train = train.pop('appname')
#     le = LabelEncoder()
#     y_train = le.fit_transform(y_train)
#     # y_train = keras.utils.to_categorical(y_train, 7)
#     newx = train.values
#     # newx = newx.reshape((len(newx),1,20))
#     ds = tf.data.Dataset.from_tensor_slices((newx, y_train))
#     ds = ds.batch(batch_size)
#
#     return ds

if __name__ == '__main__':
    opt = Options().parse()
    # task  = preTrainTask(opt,latent_dim = 18,epochs=50)
    # task.fit('./csv_data/dataframe7.csv')
    mlp = mlpTask(opt)
    mlp.train()
