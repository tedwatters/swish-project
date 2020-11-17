
'''

 Based on 
 https://www.tensorflow.org/tutorials/images/cnn
 
'''

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from collections.abc import Iterable

import numpy as np

from scipy.stats.mstats import describe

'''
Define Swish Function
'''

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

def swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

class Swish(Layer):

    def __init__(self, beta=1.0, trainable=False,  **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta,
                                      dtype=K.floatx(),
                                      name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return swish(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


'''
Get data
'''

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


'''
ReLU model
'''

r_model = models.Sequential()
r_model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
r_model.add(layers.Activation('relu'))
r_model.add(layers.MaxPooling2D((2, 2)))
r_model.add(layers.Conv2D(64, (3, 3)))
r_model.add(layers.Activation('relu'))
r_model.add(layers.MaxPooling2D((2, 2)))
r_model.add(layers.Conv2D(64, (3, 3)))
r_model.add(layers.Activation('relu'))


r_model.add(layers.Flatten())
r_model.add(layers.Dense(64))
r_model.add(layers.Activation('relu'))

r_model.add(layers.Dense(10))

r_model.summary()

r_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

r_history = r_model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

'''
Swish model
'''

s_model = models.Sequential()
s_model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
s_model.add(Swish(beta=1.0, trainable=True,name='swish1'))
s_model.add(layers.MaxPooling2D((2, 2)))
s_model.add(layers.Conv2D(64, (3, 3)))
s_model.add(Swish(beta=1.0, trainable=True,name='swish2'))
s_model.add(layers.MaxPooling2D((2, 2)))
s_model.add(layers.Conv2D(64, (3, 3)))
s_model.add(Swish(beta=1.0, trainable=True,name='swish3'))



s_model.add(layers.Flatten())
s_model.add(layers.Dense(64))
s_model.add(Swish(beta=1.0, trainable=True,name='swish4'))

s_model.add(layers.Dense(10))

s_model.summary()

s_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

s_history = s_model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

'''
Results
'''
plt.figure(0)
plt.plot(r_history.history['accuracy'], label='relu accuracy')
plt.plot(r_history.history['val_accuracy'], label = 'relu val_accuracy')
plt.plot(s_history.history['accuracy'], label='swish accuracy')
plt.plot(s_history.history['val_accuracy'], label = 'swish val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

r_test_loss, r_test_acc = r_model.evaluate(test_images,  test_labels, verbose=2)
s_test_loss, s_test_acc = s_model.evaluate(test_images,  test_labels, verbose=2)

plt.savefig('cnn_cifar_10.png', bbox_inches='tight')

  

print(r_test_acc)
print(s_test_acc)

'''
Beta values
'''

swish1_beta = []
swish2_beta = []
swish3_beta = []
swish4_beta = []

swish1_preact = []
swish2_preact = []
swish3_preact = []
swish4_preact = []

n=range(5)
for i in n:
    #reinitialize model
    s_model = models.Sequential()
    s_model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    s_model.add(Swish(beta=1.0, trainable=True,name='swish1'))
    s_model.add(layers.MaxPooling2D((2, 2)))
    s_model.add(layers.Conv2D(64, (3, 3)))
    s_model.add(Swish(beta=1.0, trainable=True,name='swish2'))
    s_model.add(layers.MaxPooling2D((2, 2)))
    s_model.add(layers.Conv2D(64, (3, 3)))
    s_model.add(Swish(beta=1.0, trainable=True,name='swish3'))
    s_model.add(layers.Flatten())
    s_model.add(layers.Dense(64))
    s_model.add(Swish(beta=1.0, trainable=True,name='swish4'))
    s_model.add(layers.Dense(10)) 
    s_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    s_history = s_model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
    
    #append results of beta and preactivations
    swish1_beta.append(s_model.get_layer(name = 'swish1').get_weights())
    swish2_beta.append(s_model.get_layer(name = 'swish2').get_weights())
    swish3_beta.append(s_model.get_layer(name = 'swish3').get_weights())
    swish4_beta.append(s_model.get_layer(name = 'swish4').get_weights())
    
    swish1_preact.append(s_model.get_layer(index = 0).get_weights()[0].tolist())
    swish2_preact.append(s_model.get_layer(index = 3).get_weights()[0].tolist())
    swish3_preact.append(s_model.get_layer(index = 6).get_weights()[0].tolist())
    swish4_preact.append(s_model.get_layer(index = 9).get_weights()[0].tolist())
    
    i += 1
    print(i)
    

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
            
bins_beta = np.arange(0,3,0.05)
bins_preact = np.arange(-2,2,0.05)

            
swish1_beta = list(flatten(swish1_beta))
swish2_beta = list(flatten(swish2_beta))
swish3_beta = list(flatten(swish3_beta))
swish4_beta = list(flatten(swish4_beta))

swish1_preact = list(flatten(swish1_preact))
swish2_preact = list(flatten(swish2_preact))
swish3_preact = list(flatten(swish3_preact))
swish4_preact = list(flatten(swish4_preact))

plt.figure(1)
plt.hist(x=swish1_beta, bins=bins_beta, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Trained Betas - Swish Layer 1')

plt.savefig('cnn_cifar_10_beta1.png', bbox_inches='tight')

plt.figure(2)
plt.hist(x=swish2_beta, bins=bins_beta, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Trained Betas - Swish Layer 2')

plt.savefig('cnn_cifar_10_beta2.png', bbox_inches='tight')

plt.figure(3)
plt.hist(x=swish3_beta, bins=bins_beta, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Trained Betas - Swish Layer 3')

plt.savefig('cnn_cifar_10_beta3.png', bbox_inches='tight')

plt.figure(4)
plt.hist(x=swish4_beta, bins=bins_beta, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Trained Betas - Swish Layer 4')

plt.savefig('cnn_cifar_10_beta4.png', bbox_inches='tight')

plt.figure(5)
plt.hist(x=swish1_preact, bins=bins_preact, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Preactivations - Swish Layer 1')

plt.savefig('cnn_cifar_10_preact1.png', bbox_inches='tight')

plt.figure(6)
plt.hist(x=swish2_preact, bins=bins_preact, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Preactivations - Swish Layer 2')

plt.savefig('cnn_cifar_10_preact2.png', bbox_inches='tight')

plt.figure(7)
plt.hist(x=swish3_preact, bins=bins_preact, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Preactivations - Swish Layer 3')

plt.savefig('cnn_cifar_10_preact3.png', bbox_inches='tight')

plt.figure(8)
plt.hist(x=swish4_preact, bins=bins_preact, alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Preactivations - Swish Layer 4')

plt.savefig('cnn_cifar_10_preact4.png', bbox_inches='tight')


print('For Swish Layer 1')
mean_value = np.mean(swish1_beta)
print('Mean of Beta: {}'.format(mean_value))
print('Minimum x where Swish is increasing: {}'.format(-1.28/np.mean(swish1_beta)))

count = 0
for i in swish1_preact: 
    if i <  mean_value: 
        count = count + 1

print('% of Preactivations less than minimum x: {}'.format(count/len(swish1_preact)))
print('Min Preactivation - Beta: {}'.format(min(swish1_preact)-mean_value))

print('For Swish Layer 2')
mean_value = np.mean(swish2_beta)
print('Mean of Beta: {}'.format(mean_value))
print('Minimum x where Swish is increasing: {}'.format(-1.28/np.mean(swish2_beta)))

count = 0
for i in swish2_preact: 
    if i <  mean_value: 
        count = count + 1

print('% of Preactivations less than minimum x: {}'.format(count/len(swish2_preact)))
print('Min Preactivation - Beta: {}'.format(min(swish2_preact)-mean_value))


print('For Swish Layer 3')
mean_value = np.mean(swish3_beta)
print('Mean of Beta: {}'.format(mean_value))
print('Minimum x where Swish is increasing: {}'.format(-1.28/np.mean(swish3_beta)))

count = 0
for i in swish3_preact: 
    if i <  mean_value: 
        count = count + 1

print('% of Preactivations less than minimum x: {}'.format(count/len(swish3_preact)))
print('Min Preactivation - Beta: {}'.format(min(swish3_preact)-mean_value))

print('For Swish Layer 4')
mean_value = np.mean(swish4_beta)
print('Mean of Beta: {}'.format(mean_value))
print('Minimum x where Swish is increasing: {}'.format(-1.28/np.mean(swish4_beta)))

count = 0
for i in swish4_preact: 
    if i <  mean_value: 
        count = count + 1

print('% of Preactivations less than minimum x: {}'.format(count/len(swish4_preact)))
print('Min Preactivation - Beta: {}'.format(min(swish4_preact)-mean_value))











