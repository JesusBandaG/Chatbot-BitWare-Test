from ChatBot import convert_menu, Instancer, Weak_grammars, Pedido, GuardarInfo, create_tokenizer
from trainFunctions import db_Connection
import numpy
from keras.models import load_model
import random

def chat(model, OrdersCollection, IntentionsCollection, labels, tokenizer):
    menu_principal = """\nChatBot: ¡Hola! 👋 Soy el Asistente Virtual de Almacenes Anfora. 🤖 🍴 \n
                        ¿Qué deseas? Escribe el número. \n

                        1. Sucursales (Horario, teléfono y ubicación) ☎️ \n
                        2. Tienda en línea 🛒 \n
                        3. Cotizaciones 💰 \n
                        4. Promociones 🔔 \n\n

                        Almacenes Anfora: Antes de visitarnos, te invitamos a conocer las medidas preventivas \n
                        que tenemos actualmente en nuestras tiendas, solo escribe 5 \n"""
    print(menu_principal)
    prevOpt = ""
    while True:
        inp = input("     Tú: ")

        if prevOpt != "":
          inp = prevOpt + inp
        
        inp = convert_menu(inp)

        # Instrucción de fin de conversación
        if inp.lower() == "salir":
          break
        
        # Predicción del tag
        results = model.predict(Instancer(inp, tokenizer))
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # Valor de la clase con mayor score
        maxscore = numpy.max(results)
        print('Score del intent: ' + str(maxscore))

        # Asignación de las respuestas posibles con base en el tag
        for tg in IntentionsCollection.find():
          if tg['tag'] == tag:
            responses = tg['responses']

        # Respuesta de la gramática débil
        weak = Weak_grammars(inp)

        if maxscore > 0.5:
          if tag == "Tienda":
            print('\nChatBot: ' + str(random.choice(responses)) + ' [' + str(tag) + ']\n')
            prevOpt = "2"
          elif tag == "Pedido":
            print('\nChatBot: ' + str(random.choice(responses)) + ' [' + str(tag) + ']\n')
            orden = input("     Tú: ")
            Pedido(inp, orden, labels, IntentionsCollection, OrdersCollection, tokenizer)
            prevOpt = ""
          elif tag == "Sucursales":
            print('\nChatBot: ' + str(random.choice(responses)) + ' [' + str(tag) + ']\n')
            tel = input("     Tú: ")
            GuardarInfo(tel)
            prevOpt = ""
          else:
            print('\nChatBot: ' + str(random.choice(responses)) + ' [' + str(tag) + ']\n')
            prevOpt = ""
        else:
          if weak == 0:
            print('\nLo siento, pero no comprendí, ¿Me puedes preguntar de otra forma?\n')
            prevOpt = ""
            print(menu_principal)


if __name__ == '__main__':
    model = load_model('Chatbot_Model.h5')
    OrdersCollection, IntentionsCollection, texts, labels = db_Connection()
    tokenizer = create_tokenizer(texts)
    chat(model, OrdersCollection, IntentionsCollection, labels, tokenizer)