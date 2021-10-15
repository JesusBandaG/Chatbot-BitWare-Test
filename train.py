from trainFunctions import db_Connection, embeddings, removeStopwords, createInputMatrix, genOutputs
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt


def train(IntentionsCollection, texts, labels):

    train_labels = genOutputs(IntentionsCollection, labels)
    X = removeStopwords(texts)
    X_train, tokenizer = createInputMatrix(X)
    vocab_size, embedding_matrix = embeddings(tokenizer)

    # Declaración de las capas del modelo LSTM
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=X_train.shape[1] , trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(labels), activation='softmax'))

    # Compilación del modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Entrenamiento
    history = model.fit(X_train, train_labels, epochs=100, batch_size=2, verbose=1)

    # Cálculo de los procentajes de Eficiencia y pérdida 
    score = model.evaluate(X_train, train_labels, verbose=1)
    print("\nTest Loss:", score[0])
    print("Test Accuracy:", score[1])

    # Guardamos el modelo
    model.save('Chatbot_Model.h5')

    # Graficamos la exactitud y la pérdida del modelo época a época para ver su evolución
    plt.figure(figsize=(12,5))
    plt.ylim(-0.1, 1.1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Acc','Loss'])
    plt.show()


if __name__ == '__main__':
    _, IntentionsCollection, texts, labels = db_Connection()
    train(IntentionsCollection, texts, labels)