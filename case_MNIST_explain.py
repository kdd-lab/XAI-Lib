import tensorflow as tf

if __name__ == '__main__':

    print(tf.__version__)

    from tensorflow.keras.datasets import mnist
    (MNIST_x_data_train, MNIST_y_data_train), (MNIST_x_data_test, MNIST_y_data_test) = mnist.load_data()

    MNIST_CNN = tf.keras.models.load_model('../models/cnn_simple_mnist_no_pickle')
    MNIST_CNN.trainable=False

    MNIST_CNN.summary()

    from xailib.explainers.intgrad_explainer import IntgradImageExplainer

    ig = IntgradImageExplainer(MNIST_CNN)

    ig.fit(1,'black')

    image = MNIST_x_data_train[0,:].reshape(28,28,1)
    scores = ig.explain(image)

    print('score shape = ',scores.shape)

    import matplotlib.pyplot as plt
    plt.imshow(scores,cmap='coolwarm')
    plt.axis('off')
    plt.show()
