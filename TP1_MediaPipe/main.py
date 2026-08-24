import cv2
import mediapipe as mp
import random
import time


# -----------------------------
# Configuración de MediaPipe
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


# -----------------------------
# Detectar dedos
# -----------------------------

def detectar_dedos(hand):

    dedos = []

    # Índice
    if hand[8].y < hand[6].y:
        dedos.append("Indice")

    # Medio
    if hand[12].y < hand[10].y:
        dedos.append("Medio")

    # Anular
    if hand[16].y < hand[14].y:
        dedos.append("Anular")

    # Meñique
    if hand[20].y < hand[18].y:
        dedos.append("Menique")

    return dedos


# -----------------------------
# Reconocer jugada
# -----------------------------

def reconocer_jugada(dedos):

    # Piedra
    if len(dedos) == 0:
        return "PIEDRA"

    # Tijera
    elif len(dedos) == 2 and "Indice" in dedos and "Medio" in dedos:
        return "TIJERA"

    # Papel
    elif len(dedos) == 4:
        return "PAPEL"

    return "NO RECONOCIDO"


# -----------------------------
# Determinar ganador
# -----------------------------

def determinar_ganador(jugador, computadora):

    if jugador == computadora:
        return "EMPATE"

    if (
        jugador == "PIEDRA" and computadora == "TIJERA"
        or jugador == "PAPEL" and computadora == "PIEDRA"
        or jugador == "TIJERA" and computadora == "PAPEL"
    ):
        return "GANASTE"

    return "PERDISTE"


# -----------------------------
# Variables del juego
# -----------------------------

jugadas = ["PIEDRA", "PAPEL", "TIJERA"]

puntaje_jugador = 0
puntaje_computadora = 0

estado = "ESPERANDO"

inicio_ronda = 0

jugada_jugador = "..."
jugada_computadora = "..."

resultado = ""


# -----------------------------
# Cámara
# -----------------------------

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("No se pudo acceder a la cámara")
            break

        # Efecto espejo
        frame = cv2.flip(frame, 1)

        # -----------------------------
        # Procesar imagen
        # -----------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # -----------------------------
        # Detectar mano
        # -----------------------------

        jugada_detectada = "NO RECONOCIDO"

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            dedos = detectar_dedos(hand)

            jugada_detectada = reconocer_jugada(dedos)


            # Dibujar puntos
            height, width, _ = frame.shape

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


        # =====================================================
        # ESTADO: ESPERANDO
        # =====================================================

        if estado == "ESPERANDO":

            cv2.putText(
                frame,
                "Presiona ESPACIO para jugar",
                (80, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


        # =====================================================
        # ESTADO: CUENTA REGRESIVA
        # =====================================================

        elif estado == "CUENTA":

            tiempo = time.time() - inicio_ronda

            restante = 3 - int(tiempo)

            if restante <= 0:

                estado = "RESULTADO"

                jugada_jugador = jugada_detectada

                jugada_computadora = random.choice(jugadas)

                if jugada_jugador != "NO RECONOCIDO":

                    resultado = determinar_ganador(
                        jugada_jugador,
                        jugada_computadora
                    )

                    if resultado == "GANASTE":
                        puntaje_jugador += 1

                    elif resultado == "PERDISTE":
                        puntaje_computadora += 1

                else:

                    resultado = "GESTO NO RECONOCIDO"

                inicio_ronda = time.time()

            else:

                cv2.putText(
                    frame,
                    str(restante),
                    (300, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 255, 255),
                    8
                )


        # =====================================================
        # ESTADO: RESULTADO
        # =====================================================

        elif estado == "RESULTADO":

            cv2.putText(
                frame,
                "VOS: " + jugada_jugador,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "PC: " + jugada_computadora,
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                resultado,
                (120, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                4
            )

            # Esperar 3 segundos

            if time.time() - inicio_ronda > 3:

                estado = "ESPERANDO"


        # -----------------------------
        # Puntaje
        # -----------------------------

        cv2.putText(
            frame,
            f"Vos: {puntaje_jugador}",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"PC: {puntaje_computadora}",
            (400, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # -----------------------------
        # Mostrar cámara
        # -----------------------------

        cv2.imshow(
            "Piedra Papel Tijera",
            frame
        )


        # -----------------------------
        # Teclas
        # -----------------------------

        key = cv2.waitKey(1) & 0xFF

        # ESPACIO → nueva ronda
        if key == 32 and estado == "ESPERANDO":

            estado = "CUENTA"
            inicio_ronda = time.time()

        # Q → salir
        elif key == ord("q"):

            break


cap.release()
cv2.destroyAllWindows()