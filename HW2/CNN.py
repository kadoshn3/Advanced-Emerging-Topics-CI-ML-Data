from d2l import tensorflow as d2l
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
def net(activation):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation=activation,
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation=activation),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation=activation),
        tf.keras.layers.Dense(84, activation=activation),
        tf.keras.layers.Dense(10)])

class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[0, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
        self.loss_arr = []
        self.train_acc_arr = []
        self.test_acc_arr = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
        
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(self.test_iter, verbose=0)[1]
        print('Epoch '+str(epoch+1)+': '+str(test_acc*100)[:4]+'%')
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.loss_arr.append(logs['loss'])
        self.train_acc_arr.append(logs['accuracy'])
        self.test_acc_arr.append(test_acc)
        self.animator.add(epoch+1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(activation, net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn(activation)
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return callback.loss_arr, callback.train_acc_arr, callback.test_acc_arr, net

if __name__ == "__main__": 
    X = tf.random.uniform((1, 28, 28, 1))
    activation = 'tanh'
    for layer in net(activation).layers:
        X = layer(X)
        #print(layer.__class__.__name__, 'output shape: \t', X.shape)
    batch_size = 64
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.2, 20
    loss_tanh, train_acc_tanh, test_acc_tanh, net_ = train_ch6(activation, net, train_iter, test_iter, num_epochs, lr)
    
    activation = 'sigmoid'
    loss_sigmoid, train_acc_sigmoid, test_acc_sigmoid, net_ = train_ch6(activation, net, train_iter, test_iter, num_epochs, lr)
    
    activation = 'relu'
    loss_relu, train_acc_relu, test_acc_relu, net_ = train_ch6(activation, net, train_iter, test_iter, num_epochs, lr)
    
    # Plot the losses
    plt.figure()
    plt.plot(np.arange(1, num_epochs+1), loss_tanh, label='Tanh')
    plt.plot(np.arange(1, num_epochs+1), loss_sigmoid, linestyle='--', dashes=(5,10), label='Sigmoid')
    plt.plot(np.arange(1, num_epochs+1), loss_relu, linestyle='--', dashes=(5,1), label='ReLU')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss for varying Activations')
    plt.legend()
    
    
    plt.figure()
    plt.plot(np.arange(1, num_epochs+1), train_acc_tanh, label='Tanh')
    plt.plot(np.arange(1, num_epochs+1), train_acc_sigmoid, linestyle='--', dashes=(5,10), label='Sigmoid')
    plt.plot(np.arange(1, num_epochs+1), train_acc_relu, linestyle='--', dashes=(5,1), label='ReLU')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy for varying Activations')
    plt.legend()
    
    plt.figure()
    plt.plot(np.arange(1, num_epochs+1), test_acc_tanh, label='Tanh')
    plt.plot(np.arange(1, num_epochs+1), test_acc_sigmoid, linestyle='--', dashes=(5,10), label='Sigmoid')
    plt.plot(np.arange(1, num_epochs+1), test_acc_relu, linestyle='--', dashes=(5,1), label='ReLU')
    plt.xlabel('Epochs')
    plt.ylabel('Testing Accuracy')
    plt.title('Testing Accuracy for varying Activations')
    plt.legend()
    
    
    