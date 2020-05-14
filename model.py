from keras import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Reshape, LSTM, Dropout
from keras.models import model_from_json
from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers


# Attention GRU network
class Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def build(n_class):
    model = Sequential()
    model.add(
        Conv2D(filters=128, input_shape=(300, 40, 3), activation='relu', padding='same', kernel_size=(5, 3), strides=1))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(filters=512, activation='relu', padding='same', kernel_size=(5, 3), strides=1))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Reshape((-1, 5 * 512)))
    model.add(Dense(786, activation='relu'))
    model.add(Reshape((150, 786)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Attention(128))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # model.add(Reshape((None, 150, 786)))
    # model.add(BatchNormalization())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def load_model():
    # 加载json
    model_path = 'model/model.h5'
    model_json_path = 'model/model.json'

    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # 加载权重
    model.load_weights(model_path)
    return model


def save_model(model):
    h5_save_path = 'model/model.h5'
    model.save_weights(h5_save_path)

    save_json_path = 'model/model.json'
    with open(save_json_path, "w") as json_file:
        json_file.write(model.to_json())


#build(4)
