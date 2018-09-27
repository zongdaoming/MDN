import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

def seq_network():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model
    # model.fit(x_train, y_train, batch_size=32, epochs=10)
    # score = model.evaluate(x_test, y_test, batch_size=32)

def fun_network():
    inputs = Input(shape=(100,100,3))
    layer_1 = Conv2D(32,(3,3),padding='same',activation='relu')(inputs)
    layer_2 = Conv2D(32,(3,3),padding='same',activation='relu')(layer_1)
    layer_3 = MaxPooling2D((2,2),strides=(1,1),padding='same')(layer_2)
    layer_4 = Dropout(0.25)(layer_3)

    layer_5 = Conv2D(64,(3,3),activation='relu')(layer_4)
    layer_6 = Conv2D(64,(3,3),activation='relu')(layer_5)
    layer_7 = MaxPooling2D((2,2),strides=(1,1),padding='same')(layer_6)
    layer_8 = Dropout(0.25)(layer_7)

    layer_9 = Flatten()(layer_8)
    layer_10 = Dense(256,activation='relu')(layer_9)
    layer_11 = Dropout(0.5)(layer_10)
    predictions = Dense(10,activation='softmax')(layer_11)

    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    # model.fit(x_train,y_train,batch_size=32,epochs=10)
    # score = model.evaluate(x_test,y_test,batch_size=32)
    return model


model = fun_network()

plot_model(model,to_file='test.pdf',show_shapes=True,show_layer_names=True)

