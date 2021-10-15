from keras.preprocessing.text import Tokenizer
from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import random
import datetime
import numpy



def create_tokenizer(texts):
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
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    return tokenizer

def Instancer(inp, tokenizer):
    """Convierte el texto de entrada en la secuencia de 
    valores enteros con pad_sequences, elimina signos
    de interrogación y acentos"""
    inp = inp.lower()
    inp = inp.replace("á", "a")
    inp = inp.replace("é", "e")
    inp = inp.replace("í", "i")
    inp = inp.replace("ó", "o")
    inp = inp.replace("ú", "u")
    inp = inp.replace("¿", "")
    inp = inp.replace("?", "")
    txt = [inp]
    maxlen_user = 2
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=maxlen_user)
    return padded

def Weak_grammars(inp):
    """Módulo de detección de gramáticas débiles"""
    Saludos_In = ["Hola", "Holi", "Cómo estás", "Que tal", "Cómo te va"]
    Despedidas_In = ["Adios", "Bye", "Hasta luego", "Nos vemos", "Hasta pronto"]
    Gracias_In = ["gracias", "te agradezco", "te doy las gracias"]
    InsD = [Saludos_In, Despedidas_In, Gracias_In]

    Saludos_Out = ["Hola ¿Cómo estás?", "Es un gusto saludarte de nuevo", "Me da gusto verte de nuevo"]
    Despedidas_Out = ["Nos vemos, fue un gusto", "Que te vaya muy bien", "Regresa pronto, adios"]
    Gracias_Out = ["Por nada, es un placer", "Me da mucho gusto poder ayudar", "De nada, para eso estoy"]
    OutsD = [Saludos_Out, Despedidas_Out, Gracias_Out]

    index = 0
    weak_act = 0

    for categoria in InsD:
        for gramatica in categoria:
            if inp.lower().count(gramatica.lower()) > 0:
                weak_act = 1
                print('\nChatBot: ' + random.choice(OutsD[index]) + ' [Gramática Débil]\n')
        index += 1
    return weak_act

def convert_menu(inp):
    """Módulo de reconocimiento de selección del menú"""
    menu = {
        '1': 'info sucursales',
        '2': 'tienda linea',
        '3': 'cotizaciones',
        '4': 'promociones',
        '5': 'infografia coronavirus',
        '21': 'Comprar',
        "22": "rastrear pedido",
        "23": "problema pedido",
        "24": "cancelar pedido"
        }
    resp = inp

    if inp.isdigit():
        try:
            resp = menu[inp]
        except:
            resp = inp

    return resp


def StatusPedido(fecha, orden, OrdersCollection):
    """Módulo de obtención de status de pedido"""
    fecha_nac = datetime.datetime.strptime(fecha, '%d/%m/%y')
    query = { "$and": [ { "order.order_number": orden }, { "customer_info.birthdate": fecha_nac } ] }
    result = OrdersCollection.find(query)
    status = ''
    for item in result:
        status = item['order']['status']
        if status == 'Pedido listo':
            guia = item['order']['tracking_guide']
            mensaje = f"\nChatBot: Tu pedido ya está listo. 👇 \nGuía de seguimiento:{guia} \n\n¡Gracias por utilizar este servicio!"
        elif status == 'En proceso':
            mensaje = "\nChatBot: A tu pedido le falta un poco más de tiempo, ten paciencia, por favor. \n¡Vuelve a revisar más tarde! 🤗 \n\n¡Gracias por utilizar este servicio!"
        else:
            mensaje = '\nChatBot: La fecha de nacimiento y/o el No. de pedido no coinciden. \n¡Vuelve a intentarlo! 🤗 \n\n¡Gracias por utilizar este servicio!'

    return mensaje


def Pedido(inp, orden, labels, IntentionsCollection, OrdersCollection, tokenizer):
    """Módulo para ejecutar acciones de acuerdo con la petición referente al pedido"""
    model = load_model("Chatbot_Model.h5")
    results = model.predict(Instancer(orden, tokenizer))
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in IntentionsCollection.find():
      if tg['tag'] == tag:
        responses = tg['responses']

    if tag == 'Operador':
      print('\nChatBot: ' + str(random.choice(responses)) + ' [' + str(tag) + ']\n')
    else:
      if inp == "rastrear pedido":
        print("\nChatBot: Por último, escribe tu fecha de nacimiento, sigue mi ejemplo con formato (dd/mm/aa): 23/02/87")
        fecha_nac = input("\n     Tú: ")
        print("\nChatBot: Hemos recibido tu fecha de nacimiento, estamos buscando tu pedido 🔎 \n¡Espera un momento!")
        mensaje = StatusPedido(fecha_nac, orden, OrdersCollection)
        print(mensaje)
        
      elif inp == "problema pedido" or "cancelar pedido":
        print("Contactando con agente en Cedis...")

def GuardarInfo(tel):
  print("\nChatBot: Por favor compárteme tu correo electrónico")
  email = input("     Tú: ")
  print("\nChatBot: ¡Perfecto! 👏, Prepárate 📝 \n💡 TIP: Puedes mandar la foto de tu lista con el nombre de cada artículo y cantidad que necesitas (piezas)")
  print("\nChatBot: Un agente tomará tu pedido ¿Estás listo? Si/No \nTambién puedes escribir Menú para ver otras opciones. 🤖")
  resp = input("     Tú: ")
  print("\nChatBot: En seguida te contactaré con un agente de Ventas...")