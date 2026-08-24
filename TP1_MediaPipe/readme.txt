Cómo usarlo:
* Tener una cámara activada
* Entrar a CMD y ejecutar python main.py
* Se tiene que abrir una ventana con la cámara y los puntos que detecta de la mano.
* Para cerrarlo apreta la Q.

******************************************************************************************

Python

Es el lenguaje en el que desarrollamos toda la aplicación.

Lo usamos para:

Procesar los datos de la cámara.
Ejecutar MediaPipe.
Analizar los dedos.
Determinar Piedra, Papel o Tijera.
Generar la jugada aleatoria de la computadora.
Calcular quién gana.
Llevar el puntaje.


MediaPipe

Es la tecnología principal del TP.

Utilizamos específicamente MediaPipe Hand Landmarker, que permite detectar una mano en una imagen o video y obtener 21 puntos de referencia (landmarks) de la mano.



OpenCV

Utilizamos OpenCV para trabajar con la cámara.

Se encarga principalmente de:

Abrir la cámara web.
Capturar los frames.
Convertir las imágenes al formato que necesita MediaPipe.
Mostrar el video.
Dibujar los puntos sobre la mano.
Mostrar textos como PIEDRA, PAPEL, GANASTE, etc.