import keras

from keras import Sequential

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.initializers import HeNormal
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization

def build_model():
    """Creates and returns a model described above

    Returns
    -------
    keras.Model
        The Sequential model created
    """

    model = Sequential()

    # -------------------------------------Abstraction path:-------------------------------------
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal(), input_shape=(64,64,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    

    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    # -------------------------------------Bottleneck path:-------------------------------------
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # -------------------------------------Expansion path:-------------------------------------
    model.add(UpSampling2D((2,2), interpolation="bilinear"))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    

    # -------------------------------------Final layer (FIXED):-------------------------------------
    model.add(Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid'))

    model.build((None, 64,64,1))

    return model

def compile_model(model, lr):
    """Compile a model according to the description above

    Parameters
    ----------
    model : keras.Model
        The model to compile
    lr : float
        The learning ratea

    Returns
    -------
    """
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')


def train_model(model, epochs, x_train, y_train, x_val, y_val):
    """Train the model according to the description above.
    NOTE: Please return your `model.fit(...)` call
    for us to grade.

    Parameters
    ----------
    model : keras.Model
        The model to train
    epochs : int
        The number of epochs to train
    x_train : np.ndarray
    y_train : np.ndarray
    x_val : np.ndarray
    y_val : np.ndarray

    Returns
    -------
    """

    model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_data=(x_val, y_val), validation_batch_size=16)


def predict_model(model, x_val):
    """Predict the segmentation masks of the validation data.

    Parameters
    ----------
    model : keras.Model
        The model to evaluate
    x_val : np.ndarray

  
    Returns
    ----------
    np.ndarray
        Predicted segmentation array on validation data
    """

    val_preds = model.predict(x_val, batch_size=16).reshape(x_val.shape)
    
    return val_preds