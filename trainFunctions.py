import certifi
from decouple import config
from pymongo import MongoClient
from keras.utils.np_utils import to_categorical 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import re
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray
from numpy import zeros


def db_Connection():
    ca = certifi.where()
    MONGODB_USERNAME = config('USER')
    MONGODB_PWD = config('PASSWORD')
    client = MongoClient('mongodb+srv://cluster0.xphgr.mongodb.net/ChatBot-Data',
                        username=MONGODB_USERNAME,
                        password=MONGODB_PWD,
                        tlsCAFile=ca)
    db = client["ChatBot-Data"]

    OrdersCollection = db["Orders"]
    IntentionsCollection = db["Intentions"]

    labels = []
    texts = []

    # Recopilación de textos para cada clase 
    for intention in IntentionsCollection.find():
        for pattern in intention['patterns']:
            texts.append(pattern)

        # Creamos una lista con los nombres de las clases
        if intention['tag'] not in labels:
            labels.append(intention['tag'])
    
    return OrdersCollection, IntentionsCollection, texts, labels

def genOutputs(IntentionsCollection, labels):
    """Generación de vector de Salidas a partir de tags"""
    output = []

    for intention in IntentionsCollection.find():
        for pattern in intention['patterns']:
            output.append(labels.index(intention['tag']))

    train_labels = to_categorical(output, num_classes=len(labels))
    
    return train_labels

def removeStopwords(texts):
    """Eliminación de stopwords"""

    stop_words = stopwords.words('spanish')
    X = []
    for sen in texts:
        
        sentence = sen
        # Filtrado de stopword
        for stopword in stop_words:
            sentence = sentence.replace(" " + stopword + " ", " ")
        sentence = sentence.replace("á", "a")
        sentence = sentence.replace("é", "e")
        sentence = sentence.replace("í", "i")
        sentence = sentence.replace("ó", "o")
        sentence = sentence.replace("ú", "u")
                
        # Remover espacios múltiples
        sentence = re.sub(r'\s+', ' ', sentence)
        # Convertir todo a minúsculas
        sentence = sentence.lower()
        # Filtrado de signos de puntuación
        tokenizer = RegexpTokenizer(r'\w+')
        # Tokenización del resultado
        result = tokenizer.tokenize(sentence)
        # Agregar al arreglo los textos "destokenizados" (Como texto nuevamente)
        X.append(TreebankWordDetokenizer().detokenize(result))
    
    return X


def createInputMatrix(X):
    # Cantidad de palabras máximas por vector de entrada
    maxlen_user = 2

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    # Transforma cada texto en una secuencia de valores enteros
    X_seq = tokenizer.texts_to_sequences(X)

    # Especificamos la matriz
    X_train = pad_sequences(X_seq, truncating='post', maxlen=maxlen_user)

    return X_train, tokenizer

def embeddings(tokenizer):
    # Lectura del archivo de embeddings
    embeddings_dictionary = dict()
    Embeddings_file = open('Word2Vect_Spanish.txt', encoding="utf8")

    # Extraemos las características del archivo de embeddings y las agregamos a un diccionario
    for linea in Embeddings_file:
        caracts = linea.split()
        palabra = caracts[0]
        vector = asarray(caracts[1:], dtype='float32')
        embeddings_dictionary [palabra] = vector
    Embeddings_file.close()

    vocab_size = len(tokenizer.word_index) + 1

    # Generamos la matriz de embeddings con 300 características
    embedding_matrix = zeros((vocab_size, 300))
    for word, index in tokenizer.word_index.items():
        # Extraemos el vector de embedding para cada palabra
        embedding_vector = embeddings_dictionary.get(word)
        # Si la palabra si existía en el vocabulario agregamos su vector de embeddings en la matriz
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return vocab_size, embedding_matrix