"""
Pipeline de vision del TP2: deteccion y clasificacion de lentes, lapicera y
moneda.

Este modulo es puro procesamiento de imagenes: no abre la camara, no crea
ventanas y no lee trackbars. Recibe arrays de numpy y devuelve arrays y
estructuras de datos. Toda la parte de interfaz vive en main.py.

Gracias a eso el pipeline se puede probar sin webcam (ver test_vision.py).
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np


# -----------------------------
# Constantes del sistema
# -----------------------------

NOMBRES = ["lentes", "lapicera", "moneda"]

DESCONOCIDO = "DESCONOCIDO"

# Un color por clase, y rojo para lo que no se reconoce.
COLORES = {
    "lentes": (0, 255, 0),        # verde
    "lapicera": (255, 255, 0),    # cian
    "moneda": (0, 215, 255),      # amarillo
    DESCONOCIDO: (0, 0, 255),     # rojo
}

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")

# Un contorno que ocupa mas de esta fraccion del cuadro no es un objeto:
# es el fondo que quedo mal binarizado.
FRACCION_MAXIMA = 0.60

# cv2.matchShapes necesita un contorno con forma; los de 3 o 4 puntos son ruido.
PUNTOS_MINIMOS = 5

FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Por debajo de este brillo medio un frame no tiene imagen. El umbral es bajo a
# proposito: una habitacion con poca luz da un brillo medio de 25 o 30, y eso no
# es una camara bloqueada; un frame bloqueado da menos de 1.
BRILLO_MINIMO = 5

AVISO_CAMARA_NEGRA = ("la camara entrega negro: cerra la app Camara de Windows, "
                      "Zoom, Teams o el navegador, o destapa el lente")

# Metrica de comparacion de formas. cv2.matchShapes ofrece tres, todas sobre los
# momentos de Hu (m = log con signo del momento):
#
#   I1 = suma |1/mA - 1/mB|      I2 = suma |mA - mB|      I3 = max |mA - mB|/|mA|
#
# Se usa I3. Medido con las fotos de referencia reales, distancia de cada objeto
# contra su propia referencia:
#
#                        I2                I3
#   escala 1.0 a 0.45    hasta 1.39        hasta 0.39
#   rotado en el plano   hasta 0.29        hasta 0.29
#   inclinado (lentes)   0.49 a 0.75       0.09 a 0.13
#   inclinado (moneda)   0.36 a 0.78       0.09 a 0.17
#
# I3 es entre 5 y 6 veces mas estable frente a la inclinacion, que es la
# variacion que de verdad aparece en vivo: el objeto casi nunca esta apoyado
# igual que en la foto. La razon es que I2 SUMA las siete diferencias, asi que
# acumula el error de detalle entre una referencia sacada de cerca y un objeto
# que se ve mas chico; I3 toma el maximo relativo y no acumula.
#
# I1 queda descartada: divide por el momento, y en objetos alargados los
# momentos de orden alto tienden a cero. Con I1 una barra contra su propia
# referencia da 27.6 y una barra rotada 30 grados se clasifica como moneda.
#
# Lo que I3 NO resuelve: la lapicera muy inclinada. Al ser tan alargada, su
# denominador tambien tiende a cero y la distancia se dispara (16 o mas con una
# inclinacion fuerte). Es un caso duro para cualquier metodo basado en momentos
# de Hu, y en la practica una lapicera apoyada en la mesa gira en el plano
# (donde I3 da 0.29 como mucho) mucho mas de lo que se inclina.
METRICA = cv2.CONTOURS_MATCH_I3

# Tamano del elemento estructural con el que arranca la morfologia. Es chico a
# proposito: la apertura y el cierre son operaciones de tamano FIJO, asi que
# cuanto mas grande el elemento, mas deforman la silueta de un objeto que se ve
# chico en el cuadro, y mas sube la distancia de matchShapes sin que el objeto
# haya cambiado. Medido sobre las fotos de referencia reales, sumando la peor
# distancia de las tres clases a lo largo de varias escalas y rotaciones:
#
#   k=0 (1x1)  2.17     k=2 (5x5)  5.86
#   k=1 (3x3)  5.10     k=3 (7x7)  6.34
#
# k=0 seria el mas fiel pero no limpia nada, y la consigna pide morfologia.
# k=1 saca el ruido suelto deformando lo minimo.
#
# Este mismo valor lo usa contorno_principal() para extraer las referencias: si
# la referencia y la deteccion se limpian con elementos distintos, las dos
# siluetas quedan deformadas distinto y la distancia sube sin motivo.
KERNEL_DEFECTO = 1

# Distancia maxima de validez con la que arranca cada clase. Son distintas a
# proposito, y son el unico lugar donde estos valores estan escritos: main.py
# inicializa sus barras a partir de aca.
#
# Estan calibrados sobre fotos reales, no sobre formas ideales. La primera
# version se calibro con circulos y rectangulos perfectos comparados contra si
# mismos, y daba valores (0.05 / 0.15 / 0.25) que en la practica rechazaban casi
# todo: en vivo el objeto nunca esta apoyado exactamente igual que en la foto de
# referencia, y matchShapes es invariante a la escala y a la rotacion en el
# plano, pero NO a la inclinacion.
#
# Lo medido con las fotos de referencia reales y la metrica I3, distancia del
# objeto contra su propia referencia frente a la distancia de formas que no son
# ese objeto (una mano, un triangulo, un cuadrado, una L, un celular, una llave):
#
#   clase      escala   rotado   inclinado   forma ajena    umbral
#   lentes      0.057    0.010     0.128       0.372 o mas   0.25
#   lapicera    0.392    0.294     0.695       1.110 o mas   0.75
#   moneda      0.069    0.014     0.171       0.200 o mas   0.19
#
# La moneda es la clase debil, y su margen es el mas chico: el circulo es la
# forma a la que se parece cualquier mancha compacta en momentos de Hu. Su
# umbral es el primero que conviene bajar si aparecen falsos positivos.
UMBRALES_DEFECTO = {
    "lentes": 0.25,
    "lapicera": 0.75,
    "moneda": 0.19,
}


# -----------------------------
# Estructuras de datos
# -----------------------------

@dataclass
class Parametros:
    """Todo lo que el usuario ajusta con las barras de desplazamiento."""

    umbral: int = 127
    usar_otsu: bool = True
    invertir: bool = True
    kernel: int = KERNEL_DEFECTO
    area_min: int = 500
    umbrales: dict = field(default_factory=lambda: dict(UMBRALES_DEFECTO))


@dataclass
class Deteccion:
    """Un contorno ya clasificado."""

    contorno: np.ndarray
    rect: tuple
    nombre: str
    distancia: float


@dataclass
class Resultado:
    """Salida completa de un frame, con los pasos intermedios."""

    gris: np.ndarray
    binaria: np.ndarray
    umbral_aplicado: int
    detecciones: list


# -----------------------------
# Estado de la camara
# -----------------------------

def parece_negra(frame):
    """
    Si el frame no tiene practicamente nada de luz.

    Pasa cuando otra aplicacion tiene tomada la webcam (la app Camara de
    Windows, Zoom, Teams, el navegador): Windows deja abrir el dispositivo y
    entregar frames, pero vienen todos en negro. Sin este chequeo los programas
    muestran una ventana negra sin ninguna explicacion, que es exactamente lo
    que paso en uso real.
    """

    return bool(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < BRILLO_MINIMO)


# -----------------------------
# Paso 1: escala de grises
# -----------------------------

def a_gris(frame):
    """Convierte a monocromatico y suaviza para que el threshold no pique."""

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gris, (5, 5), 0)


# -----------------------------
# Paso 2: binarizacion
# -----------------------------

def binarizar(gris, umbral, usar_otsu=False, invertir=True):
    """
    Aplica el threshold.

    Con usar_otsu=True el umbral lo calcula OpenCV solo (ajuste automatico) y
    se ignora el valor manual. Con invertir=True el objeto es lo mas oscuro que
    el umbral, que es el caso tipico de un objeto apoyado sobre una hoja
    blanca.

    Devuelve la imagen binaria y el umbral que realmente se uso, para poder
    mostrarlo en pantalla cuando lo eligio Otsu.
    """

    modo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY

    if usar_otsu:
        umbral_aplicado, binaria = cv2.threshold(gris, 0, 255, modo | cv2.THRESH_OTSU)
    else:
        umbral_aplicado, binaria = cv2.threshold(gris, umbral, 255, modo)

    return binaria, int(round(umbral_aplicado))


# -----------------------------
# Paso 3: morfologia
# -----------------------------

def limpiar(binaria, k):
    """
    Saca el ruido de la imagen binaria.

    El elemento estructural es una elipse de lado 2k+1, asi que la barra de
    desplazamiento mueve k y el tamano siempre queda impar. Primero una
    apertura (borra los puntos sueltos) y despues un cierre (tapa los agujeros
    que quedan dentro de los objetos).
    """

    if k <= 0:
        return binaria

    lado = 2 * k + 1
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))

    abierta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, elemento)
    return cv2.morphologyEx(abierta, cv2.MORPH_CLOSE, elemento)


# -----------------------------
# Paso 4 y 5: contornos y filtrado previo
# -----------------------------

def buscar_contornos(binaria, area_min):
    """
    Devuelve los contornos de objeto que vale la pena analizar.

    Se usa RETR_CCOMP y no RETR_EXTERNAL. RETR_EXTERNAL devuelve solo los
    contornos mas externos e ignora todo lo que este anidado adentro, y eso
    rompia el caso mas comun de uso: una hoja blanca apoyada sobre un escritorio
    mas oscuro que se ve alrededor. Ahi Otsu termina separando escritorio de
    hoja, no objeto de hoja, con lo cual el escritorio queda como primer plano y
    la hoja queda como un agujero adentro. Los objetos apoyados sobre la hoja
    quedan anidados dentro de ese agujero y RETR_EXTERNAL los descartaba: no se
    detectaba nada.

    RETR_CCOMP arma dos niveles: los bordes externos de cada region de primer
    plano y, abajo, los agujeros. Un objeto que esta adentro de un agujero
    igual va al primer nivel, que es justo lo que hace falta. Los agujeros
    (los que tienen padre) se descartan, para no confundir el borde de la hoja
    con un objeto.

    Ademas descarta de antemano:
      - los que tienen menos de PUNTOS_MINIMOS puntos (ruido),
      - los mas chicos que area_min (basura, sombras, granos),
      - los que ocupan mas del 60% del cuadro (el fondo mal binarizado).
    """

    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is None:
        return []

    alto, ancho = binaria.shape[:2]
    area_max = alto * ancho * FRACCION_MAXIMA

    validos = []
    for contorno, datos in zip(contornos, jerarquia[0]):
        # datos[3] es el indice del contorno padre: si tiene padre es un
        # agujero del fondo, no un objeto.
        if datos[3] != -1:
            continue

        if len(contorno) < PUNTOS_MINIMOS:
            continue

        area = cv2.contourArea(contorno)
        if area < area_min or area > area_max:
            continue

        validos.append(contorno)

    return validos


# -----------------------------
# Paso 6: clasificacion con matchShapes
# -----------------------------

def clasificar(contorno, referencias, umbrales):
    """
    Compara el contorno contra todas las referencias con cv2.matchShapes().

    Una referencia es candidata valida si su distancia no supera el umbral de
    esa clase. Gana la candidata con menor distancia. Si ninguna entra dentro
    de su umbral, la forma es DESCONOCIDO (igual se devuelve la distancia mas
    chica encontrada, que sirve para calibrar las barras).
    """

    distancia_minima = None
    ganador = None
    distancia_ganadora = None

    for nombre, referencia in referencias.items():
        distancia = cv2.matchShapes(contorno, referencia, METRICA, 0.0)

        if distancia_minima is None or distancia < distancia_minima:
            distancia_minima = distancia

        if distancia <= umbrales.get(nombre, 0.0):
            if distancia_ganadora is None or distancia < distancia_ganadora:
                ganador = nombre
                distancia_ganadora = distancia

    if ganador is None:
        return DESCONOCIDO, distancia_minima

    return ganador, distancia_ganadora


# -----------------------------
# Pipeline completo
# -----------------------------

def procesar(frame, params, referencias):
    """Corre todo el pipeline sobre un frame y devuelve tambien los pasos."""

    gris = a_gris(frame)
    binaria, umbral_aplicado = binarizar(gris, params.umbral, params.usar_otsu, params.invertir)
    binaria = limpiar(binaria, params.kernel)

    detecciones = []
    for contorno in buscar_contornos(binaria, params.area_min):
        nombre, distancia = clasificar(contorno, referencias, params.umbrales)
        detecciones.append(
            Deteccion(
                contorno=contorno,
                rect=cv2.boundingRect(contorno),
                nombre=nombre,
                distancia=distancia,
            )
        )

    return Resultado(gris, binaria, umbral_aplicado, detecciones)


# -----------------------------
# Paso 7: anotacion
# -----------------------------

def anotar(frame, detecciones):
    """Dibuja contorno, rectangulo y etiqueta sobre una copia del frame."""

    salida = frame.copy()

    for deteccion in detecciones:
        color = COLORES.get(deteccion.nombre, COLORES[DESCONOCIDO])
        x, y, w, h = deteccion.rect

        cv2.drawContours(salida, [deteccion.contorno], -1, color, 2)
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 1)

        texto = deteccion.nombre.upper()
        if deteccion.distancia is not None:
            texto += f" ({deteccion.distancia:.3f})"

        _dibujar_etiqueta(salida, texto, x, y, color)

    return salida


def _dibujar_etiqueta(imagen, texto, x, y, color):
    """Etiqueta con fondo del color de la clase, para que se lea siempre."""

    (ancho_texto, alto_texto), base = cv2.getTextSize(texto, FUENTE, 0.5, 1)

    y_base = max(y - 6, alto_texto + 6)
    esquina1 = (x, y_base - alto_texto - 5)
    esquina2 = (x + ancho_texto + 8, y_base + base - 2)

    cv2.rectangle(imagen, esquina1, esquina2, color, -1)
    cv2.putText(imagen, texto, (x + 4, y_base - 2), FUENTE, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# -----------------------------
# Referencias (fotos de la carpeta referencias/)
# -----------------------------

def fondo_es_claro(gris, margen=5):
    """
    Mira el marco exterior de la imagen para saber sobre que fondo esta el
    objeto.

    Se asume que el objeto no toca los bordes de la foto de referencia, asi que
    el marco es fondo puro. Si el marco es claro, el objeto hay que buscarlo en
    lo oscuro (y al reves). Esto evita tener que sacar todas las fotos con el
    mismo tipo de fondo.

    La comparacion es contra el umbral de Otsu de la propia imagen, no contra un
    valor fijo. Antes se comparaba contra 127 y eso fallaba con cualquier foto
    sacada con poca luz: una hoja blanca subexpuesta sale gris (~100), o sea por
    debajo de 127, y el programa decidia que el fondo era oscuro. Con la
    polaridad invertida el "objeto" pasaba a ser todo el cuadro menos el objeto,
    y la referencia quedaba inservible.

    Se usa la mediana del marco y no el promedio, para que una sombra o una
    esquina oscura no arrastren la decision.
    """

    umbral, _ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    borde = np.concatenate([
        gris[:margen, :].ravel(),
        gris[-margen:, :].ravel(),
        gris[:, :margen].ravel(),
        gris[:, -margen:].ravel(),
    ])

    return bool(np.median(borde) > umbral)


def toca_el_borde(contorno, forma, margen=2):
    """Si la caja del contorno llega a algun borde de la imagen."""

    alto, ancho = forma[:2]
    x, y, w, h = cv2.boundingRect(contorno)

    return x <= margen or y <= margen or x + w >= ancho - margen or y + h >= alto - margen


def contorno_principal(imagen):
    """
    Extrae el contorno del objeto de una foto de referencia.

    Binariza con Otsu, eligiendo la polaridad sola segun el fondo, limpia con
    morfologia y elige un contorno. Devuelve None si no encuentra ninguno
    usable, para que main.py pueda avisar en vez de cargar una referencia
    inservible en silencio.

    Entre los candidatos se prefieren los que NO tocan los bordes de la imagen,
    y recien entre esos se toma el de mayor area. El motivo: una sombra, o el
    borde del escritorio asomando en una esquina, puede ser mas grande que el
    objeto y ganarle por area. Ademas es la misma suposicion que ya hace
    fondo_es_claro, que da por sentado que el marco de la foto es fondo puro.

    Si no queda ninguno despegado del borde se usa el mas grande igual, porque
    un objeto largo como la lapicera puede cruzar todo el cuadro.
    """

    gris = a_gris(imagen)
    binaria, _ = binarizar(gris, umbral=0, usar_otsu=True, invertir=fondo_es_claro(gris))
    binaria = limpiar(binaria, k=KERNEL_DEFECTO)

    # area_min=0 porque aca interesa el objeto mas grande, sea cual sea su
    # tamano; lo que si aporta buscar_contornos es descartar los agujeros y el
    # contorno que ocupa casi todo el cuadro (una foto asi no sirve).
    contornos = buscar_contornos(binaria, area_min=0)
    if not contornos:
        return None

    despegados = [c for c in contornos if not toca_el_borde(c, binaria.shape)]

    return max(despegados or contornos, key=cv2.contourArea)


def _buscar_foto(carpeta, nombre):
    """Busca nombre.jpg / .jpeg / .png / .bmp dentro de la carpeta."""

    for extension in EXTENSIONES:
        ruta = os.path.join(carpeta, nombre + extension)
        if os.path.isfile(ruta):
            return ruta

    return None


def cargar_referencias(carpeta):
    """
    Carga el contorno de referencia de cada objeto desde carpeta/.

    Devuelve tres cosas:
      - un diccionario nombre -> contorno con las que pudo cargar,
      - la lista de las que faltan,
      - una lista de avisos sobre las que se cargaron pero pintan mal.

    Los avisos importan porque una referencia mala no rompe nada: simplemente
    hace que despues no se reconozca nada, y no hay forma de darse cuenta
    mirando la ventana. El caso tipico es el objeto cortado por el borde de la
    foto: ahi se rompe la suposicion de fondo_es_claro y ademas el objeto se
    vuelve indistinguible de una sombra pegada al borde.
    """

    referencias = {}
    faltantes = []
    avisos = []

    for nombre in NOMBRES:
        ruta = _buscar_foto(carpeta, nombre)
        if ruta is None:
            faltantes.append(nombre)
            continue

        imagen = cv2.imread(ruta)
        if imagen is None:
            faltantes.append(nombre)
            avisos.append(f"{nombre}: no se pudo leer el archivo {os.path.basename(ruta)}")
            continue

        contorno = contorno_principal(imagen)
        if contorno is None:
            faltantes.append(nombre)
            avisos.append(f"{nombre}: no se pudo extraer ningun contorno de la foto")
            continue

        if toca_el_borde(contorno, imagen.shape):
            avisos.append(
                f"{nombre}: el contorno llega al borde de la foto. Puede ser el objeto "
                f"cortado, o una sombra del fondo. Conviene sacarla de nuevo."
            )

        referencias[nombre] = contorno

    return referencias, faltantes, avisos
