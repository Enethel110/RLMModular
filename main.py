from flask import Flask, request, jsonify, redirect, send_file, url_for
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

app = Flask(__name__)
global a, predictions, dia, mes, anio, model_trained, J, priori, categ, subcateg
model_trained = False
num_a_palabra = ["--",'Aloha','Asignación','BI','Cambio','CARP','Carpetas en Red','Contpaq','Delitech',
    'DVR','Facturación','Facturas','GMAIL','Infraestructura','Instalación','Mantenimiento','MBP','Plataformas','Programas Otros',
    'Protheus','Respaldo','Revisión','Servidor GIRO','Solicitud de equipo','Soporte',
    'Spoonity','WEB','WEB Server','Windows','Zetus'
]

num_a_palabra1 = ["--",
    'Archivo-Inventario','Artículos-Alta','Artículos-Baja','Articulos-Modificación', 'Artículos-Nuevo',
    'ATO-Asignar Pedido','ATO-Conexión B.D.','ATO-Consecutivo Tickets','ATO-Otros','BD-No inica','BI-caido',
    'Cambio de Hora-Factura','Cambio de Hora-Factura','Cancelación-Factura','CFC-Acceso o Permiso','CFC-Contraseña',
    'Correo-nuevo','Correo-Saturado','DRS-caído','DRS-otros','DVR-NAS','Envio-Fallido','Equipos-Impresora','Equipos-Servidor',
    'Equipos-Terminal','Impresora-Toner', 'Inventario-Ajuste','Inventario-Claves','Migrar-Sucursal','Nolo-Conexión','Otro-Duda',
    'Otro-Soporte','Pantalla-Trabada','Portal-error','Precios-Cambio', 'Precio-incorrecto','Promociones-Baja','Promociones-Modificacón',
    'Tickets-Error','Tickets-Soporte','Transferencias','Transferencias-Otros', 'Usuario-Baja','Usuario-Password',
    'Acces-Point','Audio','Antivirus','Camara','Claves-telefónicas','Equipo-sucursal'
]
prioridad = ["--","Bajo","Medio","Alto"]

# Función para entrenar el modelo y realizar predicciones
def train_and_predict(dataP):
    global a, model_trained, dia, mes, anio, J, priori, categ, subcateg
    # Validar si el modelo ya ha sido entrenado
    if not model_trained:
        # Cargar los datos
        data = pd.read_excel("datos_tickets.xlsx")

        # Convertir la columna 'Fecha' a objetos datetime y extraer el día, mes y año
        data['Fecha'] = pd.to_datetime(data['Fecha'])
        data['Dia'] = data['Fecha'].dt.day
        data['Mes'] = data['Fecha'].dt.month
        data['Año'] = data['Fecha'].dt.year

        # Eliminar la columna original 'Fecha' si no es necesaria
        data.drop(columns=['Fecha'], inplace=True)

        #Separamos los datos en X y Y
        X = data
        X = X.drop(columns=["Tickets"])
        Y = data
        Y = Y.drop(columns=["Dia", "Mes","Año","Prioridad","Categoria","Subcategoria"])
        # Desordenar los datos en Y
        Y = Y.sample(frac=1).reset_index(drop=True)

        #Agregamos columna de unos a la matriz X.
        m, n = np.shape(X)
        X.insert(0,"x0", np.ones((m,1)))    

        #Inicializamos el vector de parametros a
        a = np.ones((n+1,1))

        #Inicializar parametros
        beta = 0.00000001
        iterMax = 600

        #Crear los vectores J y h
        J = np.zeros((iterMax,1))
        h = np.zeros((m,1))

        #Entrenamiento 
        for iter in range(iterMax):
            for i in range(m):
                h[i] = np.dot(a.transpose(), X.iloc[i, :])
            J[iter] = np.sum(np.power((h-Y),2), axis=0) / (2*m)
            for j in range(n+1):
                xj = np.mat(X[X.columns[j]])
                xj = xj.transpose()
                a[j] = a[j] - beta*(1/m)*np.sum((h-Y) * xj)
        
        # Actualizar la bandera para indicar que el modelo ha sido entrenado
        model_trained = True
    # Separar la fecha en día, mes y año
    num = 1
    fecha = dataP['fecha'].split('-')
    dia = int(fecha[2])
    mes = int(fecha[1])
    anio = int(fecha[0])
    priori =int(dataP['prioridad'])
    categ = int(dataP['categoria'])
    subcateg = int(dataP['subcategoria'])
    lista_valores = [
        num,
        priori,
        categ,
        subcateg,
        dia,
        mes,
        anio
    ]
    print(lista_valores, "--------------")

    # Predicciones
    Pticket = np.dot(a.transpose(), lista_valores)
    Pticket = int(Pticket[0])
    return Pticket

@app.route('/', methods=['POST'])
def index():
    global predictions
    if request.method == 'POST':
        # Recibe los datos enviados por POST
        data = request.form

        # Realiza predicciones utilizando la función train_and_predict
        predictions = train_and_predict(data)
        print(predictions, "************+")
        # Retorna las predicciones como un diccionario JSON
        #return jsonify({'prediccion': str(predictions)})
        return redirect("https://focoesp8266.000webhostapp.com/view/PredicTicket/", code=302)
        #jsonify(predictions=str(predictions))
    else:
        return 'Método GET no permitido'

@app.route('/change_model_state', methods=['POST'])
def change_model_state():
    global model_trained
    if 'reset_model' in request.form:
        model_trained = False
        return redirect("https://focoesp8266.000webhostapp.com/view/PredicTicket/", code=302)
    return 'Solicitud no válida.'


predictions = "---"
dia = "--"
mes = "--"
anio = "--"
@app.route('/predic')
def obtener_valor():
    global predictions, dia, mes, anio
    return jsonify({'valor': predictions,
                    'dia': dia,
                    'mes': mes,
                    'año': anio})

@app.route('/statusModel')
def statusMolde():
    global model_trained
    if(model_trained == True):
        return jsonify({'trained': "TRAINED"})
    elif(model_trained == False):
        return jsonify({'trained': "NO TRAINED"})
    
priori = 0
categ = 0
subcateg = 0
@app.route('/TipTicket')
def TipTicket():
    global priori, categ, subcateg
    priori_ = prioridad[priori]
    categ_ =  num_a_palabra[categ]
    subcateg_ = num_a_palabra1[subcateg]
    return jsonify({'priori': priori_,
                    'categ': categ_,
                    'subcateg': subcateg_})
"""
@app.route('/convergencia')
def GraConvergencia():
    plt.figure()
    plt.plot(J)   
    plt.ylabel("J")
    plt.xlabel("Iteraciones")
    plt.title("Gráfica de convergencia")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image1/png')
"""
if __name__ == '__main__':
    app.run(debug=True)
