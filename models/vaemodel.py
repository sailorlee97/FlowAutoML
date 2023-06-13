import tensorflow as tf


class VAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, inputsize):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(inputsize,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(inputsize)
            ]
        )

    # 隐变量z的采样
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    # 生成网络
    # 输入：x作为观测值
    # 输出：隐变量z的条件分布(对角高斯分布的均值和对数方差)
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    # 重参数化：z = eps * std + mean, eps ~ N(0, 1)
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    # 推理网络
    # 输入：隐变量样本z
    # 输出：重构的观测值x'的条件分布(单位高斯分布)
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
