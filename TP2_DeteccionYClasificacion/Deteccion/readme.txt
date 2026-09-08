TP2 - Detección y clasificación de objetos (lentes, lapicera, moneda)


CÓMO USARLO
===========

1) Instalar las dependencias (una sola vez):

       pip install -r requirements.txt

2) Sacar las tres fotos de referencia. Con la cámara conectada:

       python capturar_referencia.py

   Poné UN solo objeto por vez frente a la cámara y apretá:
       1 -> guarda referencias/lentes.jpg
       2 -> guarda referencias/lapicera.jpg
       3 -> guarda referencias/moneda.jpg
       q -> salir

   En la ventana se ve, en verde, el contorno que el programa va a extraer de
   esa foto. Si el contorno no sigue el borde del objeto, no dispares todavía:
   cambiá el fondo o la luz hasta que lo siga bien.

   Para que las referencias salgan bien:
       - un solo objeto en el cuadro,
       - fondo liso y contrastado (una hoja blanca, o una mesa oscura),
       - el objeto entero adentro, SIN TOCAR LOS BORDES,
       - que ocupe buena parte del cuadro,
       - con luz suficiente, y sin sombras marcadas en las esquinas.

   Si el contorno sale en ROJO y dice "El contorno toca el borde", no guardes
   esa foto: o el objeto está cortado por el margen, o lo que se detectó es una
   sombra del fondo pegada al borde. Centrá el objeto y volvé a probar.

   Esto importa más de lo que parece: una referencia mala no rompe nada, sólo
   hace que después no se reconozca nada, y eso no se puede notar mirando la
   ventana. Por eso el programa avisa por consola al arrancar.

3) Correr el programa principal:

       python main.py

   Si tenés más de una cámara, se puede elegir:  python main.py 1

4) Para cerrarlo apretá la Q.


LAS VENTANAS
============

"Deteccion y Clasificacion"
    La imagen de la cámara anotada. Cada objeto sale con su contorno, un
    rectángulo que lo contiene y una etiqueta con el nombre y la distancia
    de matchShapes. Debajo de la imagen hay una franja con el umbral que se
    está aplicando, el tamaño del elemento estructural, el área mínima y
    cuántos objetos se detectaron.

    La franja va debajo y no encima de la imagen a propósito: dibujada sobre
    la imagen tapaba la etiqueta de cualquier objeto que cayera en esa
    esquina.

    Los colores:
        verde     -> lentes
        cian      -> lapicera
        amarillo  -> moneda
        rojo      -> objeto desconocido

"Pasos intermedios"
    La imagen en escala de grises y la imagen binaria (ya con las operaciones
    morfológicas aplicadas), una al lado de la otra.

Las barras de desplazamiento van en la propia ventana "Deteccion y
Clasificacion", arriba de la imagen. Antes había una tercera ventana
"Controles" sólo para eso.


LAS BARRAS DE DESPLAZAMIENTO
============================

Umbral (0-255)
    El umbral del threshold. Solo tiene efecto si "Otsu auto" está en 0.

Otsu auto (0/1)
    En 1, el umbral lo calcula OpenCV solo con el método de Otsu y se ignora
    la barra de arriba. Es el ajuste automático. En el panel siempre se ve el
    valor que terminó usando.

Invertir (0/1)
    En 1 el objeto es lo más oscuro que el umbral (objeto apoyado sobre una
    hoja blanca). En 0 es al revés. Si en la ventana "Pasos intermedios" el
    objeto sale negro y el fondo blanco, tocá esta barra.

Kernel morf (0-15)
    Tamaño del elemento estructural de las operaciones morfológicas. El
    elemento es una elipse de lado 2k+1. Primero se hace una apertura, que
    borra el ruido suelto, y después un cierre, que tapa los agujeros que
    quedan adentro de los objetos. En 0 no se aplica morfología.

    Arranca en 1 (elemento de 3x3), que es chico a propósito. La apertura y el
    cierre son operaciones de tamaño FIJO: cuanto más grande el elemento, más
    deforman la silueta de un objeto que se ve chico en el cuadro, y más sube
    la distancia de matchShapes sin que el objeto haya cambiado. Subilo sólo si
    ves ruido en la ventana "Pasos intermedios".

Area min x100 (0-200)
    Área mínima, en centenas de píxeles, para que un contorno se considere.
    Sirve para descartar de antemano contornos espúreos. En 5 el mínimo es
    500 píxeles.

Dist lentes / lapicera / moneda x100 (0-200)
    Distancia máxima de validez de cada objeto de referencia, dividida por
    100. En 25 el umbral es 0.25. Un contorno solo puede clasificarse como
    "lentes" si su distancia a la referencia de lentes no supera este valor.
    Si ningún objeto de referencia queda dentro de su umbral, la forma es
    DESCONOCIDO y se dibuja en rojo.

    Arrancan en valores distintos, y no es arbitrario (ver más abajo):

        Dist lentes   x100  ->  25   (umbral 0.25)
        Dist lapicera x100  ->  75   (umbral 0.75)
        Dist moneda   x100  ->  19   (umbral 0.19)


CÓMO CALIBRARLO
===============

La etiqueta de cada objeto muestra la distancia entre paréntesis. Ese número
es lo que hay que mirar para ajustar las barras:

    - Si un objeto que conocés sale en rojo (DESCONOCIDO), fijate la distancia
      que muestra y subí la barra de esa clase por encima de ese valor.
    - Si un objeto cualquiera se confunde con otro, bajá la barra de la clase
      equivocada.
    - Si aparecen muchos contornos de ruido, subí "Kernel morf" o
      "Area min x100".
    - Si el objeto aparece cortado o pegado al fondo, el problema es el
      threshold: probá con "Otsu auto" en 1, o tocá "Invertir".


SI LA IMAGEN SALE NEGRA
======================

Casi siempre es que OTRA APLICACIÓN tiene tomada la webcam. Windows deja que el
programa abra el dispositivo y hasta le entrega frames, pero vienen todos en
negro, sin ningún error. Los sospechosos habituales:

    - la app "Cámara" de Windows (la más común, y queda abierta sin que se note)
    - Zoom, Teams, Discord, OBS
    - una pestaña del navegador con la cámara activa

Cerralas y volvé a correr el programa. Para ver si alguna está viva, en
PowerShell:

    Get-Process WindowsCamera,Zoom,ms-teams,Discord -ErrorAction SilentlyContinue

El programa ahora avisa solo: si los frames vienen negros, muestra en rojo
"la camara entrega negro..." en la franja de datos y lo escribe por consola. No
se confunde con una habitación oscura, porque el umbral está en 5 y una
habitación con poca luz da 25 o 30.

Otras causas menos frecuentes:

    - la webcam tiene una tapita de privacidad corrida sobre el lente,
    - la tecla de función que desactiva la cámara (en algunos teclados),
    - Configuración -> Privacidad y seguridad -> Cámara, con el acceso apagado.

Nota aparte: al abrir, el programa descarta 15 frames antes de empezar. La
exposición automática de la cámara tarda en estabilizarse (medido: el primer
frame sale con brillo medio 69 y recién después de unos 25 llega a 140). Sin
eso los primeros segundos se ven oscuros, el umbral de Otsu arranca mal, y si
justo sacabas una foto de referencia te salía subexpuesta.


SI NO DETECTA NADA
==================

Mirá la ventana "Pasos intermedios". La imagen binaria tiene que mostrar los
objetos en BLANCO sobre un fondo NEGRO. Si ves lo contrario (casi todo blanco
con los objetos en negro), tocá la barra "Invertir".

El caso más común: la hoja blanca no llena todo el encuadre y se ve el
escritorio alrededor. Ahí Otsu termina separando escritorio de hoja en vez de
objeto de hoja, y el escritorio entero pasa a ser primer plano. Se ve enseguida
en la binaria: todo el borde de la imagen aparece blanco.

Soluciones, en orden:

    1. Acercá la cámara o corré la hoja hasta que la hoja llene todo el
       encuadre. Es la solución real y la más simple.
    2. Si no llega a llenarlo, al menos dejá los objetos bien adentro de la
       hoja, lejos del escritorio. El programa los detecta igual: busca objetos
       también adentro de las zonas rodeadas por el fondo.
    3. Subí "Area min x100" para que el escritorio no genere candidatos
       espúreos.

Si detecta los objetos pero salen todos en rojo (DESCONOCIDO), el problema no
es la detección sino la clasificación: fijate el número que muestra la etiqueta
y subí la barra de distancia de esa clase. Si aun así no baja, revisá que las
fotos de referencia sean buenas (el programa avisa por consola al arrancar).


TECLAS
======

    q o ESC   salir
    r         recargar las fotos de referencia (para probar fotos nuevas sin
              cerrar el programa)
    g         guardar el frame anotado en capturas/


CÓMO FUNCIONA
=============

El procesamiento de cada frame, en orden:

1. Escala de grises
   cvtColor a monocromático, más un blur gaussiano de 5x5 para que el
   threshold no quede picado.

2. Threshold
   Binarización con umbral manual o automático (Otsu), con la polaridad que
   indique la barra "Invertir".

3. Operaciones morfológicas
   Apertura seguida de cierre, con un elemento estructural elíptico de lado
   2k+1, para eliminar el ruido de la imagen.

4. Contornos
   findContours con RETR_CCOMP, quedándose con los contornos de primer nivel.
   El sistema puede obtener varios contornos en una misma imagen y los procesa
   a todos individualmente. Se usa RETR_CCOMP y no RETR_EXTERNAL porque
   RETR_EXTERNAL ignora todo lo anidado, y con una hoja que no llena el
   encuadre los objetos quedan dentro de un "agujero" del fondo (ver
   "SI NO DETECTA NADA").

5. Filtrado previo
   Se descartan de antemano los contornos con menos de 5 puntos, los de área
   menor a la mínima, y los que ocupan más del 60% del cuadro (que no son un
   objeto sino el fondo mal binarizado).

6. Clasificación
   Cada contorno se compara contra los tres objetos de referencia con
   cv2.matchShapes(). Una referencia es candidata válida si su distancia no
   supera el umbral de esa clase. Gana la candidata con menor distancia. Si no
   hay ninguna válida, la forma es DESCONOCIDO.

7. Anotación
   Se dibuja el contorno, el rectángulo contenedor y la etiqueta, con el color
   de la clase.


QUÉ MÉTRICA USA matchShapes Y POR QUÉ
=====================================

cv2.matchShapes() compara los momentos de Hu de dos contornos y acepta tres
métricas (m es el logaritmo con signo del momento):

    I1 = suma |1/mA - 1/mB|
    I2 = suma |mA - mB|
    I3 = máximo |mA - mB| / |mA|

Acá se usa I3. Medido con las fotos de referencia reales, distancia de cada
objeto contra su propia referencia:

                             I2            I3
    escala 1.0 a 0.45     hasta 1.39    hasta 0.39
    girado en el plano    hasta 0.29    hasta 0.29
    inclinado (lentes)    0.49 a 0.75   0.09 a 0.13
    inclinado (moneda)    0.36 a 0.78   0.09 a 0.17

I3 es entre 5 y 6 veces más estable frente a la inclinación, que es la
variación que de verdad aparece en vivo: el objeto casi nunca queda apoyado
igual que en la foto de referencia. La razón es que I2 SUMA las siete
diferencias, así que acumula el error de detalle entre una referencia sacada de
cerca y un objeto que se ve más chico; I3 toma el máximo relativo y no acumula.

I1 queda descartada: divide por el momento, y en objetos alargados los momentos
de orden alto tienden a cero. Con I1 una barra comparada contra su propia
referencia da 27.6, y una barra rotada 30 grados se clasifica como moneda: se
pierde la invariancia a la rotación.

Lo que I3 NO resuelve es la lapicera muy inclinada. Al ser tan alargada, su
denominador también tiende a cero y la distancia se dispara (16 o más). Es un
caso duro para cualquier método basado en momentos de Hu, y en la práctica una
lapicera apoyada en la mesa gira en el plano (donde I3 da 0.29 como máximo)
mucho más de lo que se inclina.


POR QUÉ UN UMBRAL DISTINTO PARA CADA OBJETO
===========================================

La consigna permite un umbral de distancia global o uno por objeto de
referencia. Acá hay uno por objeto, porque midiendo se ve que las tres clases
no se comparan igual de bien.

Medido con las fotos de referencia reales y la métrica I3: distancia del objeto
contra su propia referencia (cambiando tamaño, girándolo y inclinándolo) frente
a la distancia de formas que NO son ese objeto (una mano, un triángulo, un
cuadrado, una L, un celular, una llave):

    clase       escala   girado   inclinado   forma ajena    umbral
    --------    ------   ------   ---------   -----------    ------
    lentes       0.057    0.010     0.128     0.372 o más     0.25
    lapicera     0.392    0.294     0.695     1.110 o más     0.75
    moneda       0.069    0.014     0.171     0.200 o más     0.19

El umbral de cada clase tiene que quedar por encima de lo peor que da el objeto
real y por debajo de la forma ajena más cercana. La lapicera tiene lugar de
sobra; la moneda casi no tiene margen (0.171 contra 0.200), y es la primera
barra que conviene bajar si aparecen falsos positivos.

Un detalle que costó entender: la primera versión de estos umbrales se calibró
con círculos y rectángulos perfectos comparados contra sí mismos en la misma
pose, y daba 0.05 / 0.15 / 0.25. En vivo eso rechazaba casi todo, porque el
objeto nunca está apoyado igual que en la foto. Una inclinación mínima de los
lentes ya daba 0.18, por encima del 0.15 de entonces.


LIMITACIÓN CONOCIDA
===================

La forma ajena que sigue entrando es el cuadrado: su distancia a la referencia
de la moneda es 0.018, más baja incluso que la de la propia moneda vista con
otro tamaño o inclinación. No es un error del programa, es una propiedad de
los momentos de Hu: un cuadrado y un círculo son los dos compactos, convexos y
simétricos, así que sus momentos son casi iguales.

No hay ningún umbral que separe esos dos casos. Para distinguirlos habría que
mirar algo más que la forma del contorno (por ejemplo, la relación entre el
área del contorno y la de su envolvente convexa, o la cantidad de vértices al
aproximar el polígono), pero eso ya sale de lo que pide la consigna, que es
clasificar con matchShapes().


LAS FOTOS DE REFERENCIA
=======================

Cada foto de referencias/ se procesa una sola vez, al arrancar el programa:
se pasa a grises, se binariza con Otsu, se limpia con morfología y se toma el
contorno de mayor área.

La polaridad (si el objeto es lo oscuro o lo claro) la decide solo el programa:
mide el brillo del marco exterior de la foto y asume que ahí hay puro fondo.
Por eso no importa si sacaste una foto sobre fondo blanco y otra sobre fondo
negro, pero sí importa que el objeto no toque los bordes.

La comparación es contra el umbral de Otsu de la propia foto, no contra un
valor fijo. Al principio se comparaba contra 127 y eso fallaba con cualquier
foto sacada con poca luz: una hoja blanca subexpuesta sale gris (~100), o sea
por debajo de 127, así que el programa creía que el fondo era oscuro. Con la
polaridad invertida el "objeto" pasaba a ser todo el cuadro menos el objeto, y
la referencia quedaba inservible sin ningún síntoma visible.

Entre los contornos candidatos se prefieren los que NO tocan los bordes, y
recién entre esos se toma el de mayor área. Sin eso, una sombra en una esquina
puede ser más grande que el objeto y ganarle. Si ninguno está despegado del
borde se usa el más grande igual (una lapicera larga puede cruzar todo el
cuadro), pero en ese caso el programa avisa por consola.


LOS ARCHIVOS
============

main.py
    La cámara, las ventanas y las barras de desplazamiento. Nada de lógica de
    visión.

vision.py
    Todo el procesamiento de imágenes. No abre la cámara ni crea ventanas:
    recibe arrays de numpy y devuelve arrays y datos. Por eso se puede probar
    sin webcam.

capturar_referencia.py
    El ayudante para sacar las tres fotos de referencia.

test_vision.py
    Pruebas del pipeline con imágenes sintéticas. Se corren con:

        python test_vision.py

    No necesitan cámara ni las fotos de referencia. Verifican la binarización,
    la morfología, el filtrado de contornos, la clasificación (incluyendo el
    caso de la barra rotada) y la detección automática de polaridad.

referencias/
    Las tres fotos: lentes.jpg, lapicera.jpg, moneda.jpg

capturas/
    Se crea sola cuando apretás G.


TECNOLOGÍAS
===========

Python
    El lenguaje de toda la aplicación.

OpenCV
    Hace todo el trabajo de visión: abrir la cámara y capturar los frames,
    convertir a escala de grises, binarizar con threshold y Otsu, aplicar las
    operaciones morfológicas, buscar los contornos, comparar formas con
    matchShapes, dibujar las anotaciones y mostrar las ventanas con las barras
    de desplazamiento.

NumPy
    Las imágenes de OpenCV son arrays de NumPy. Se usa para armar las vistas
    combinadas, el panel semitransparente y las imágenes sintéticas de las
    pruebas.
