"""
Pruebas del pipeline de vision, sin camara.

Se ejecutan con:   python test_vision.py

Todas las imagenes son sinteticas (generadas con numpy y OpenCV), asi que las
pruebas corren en cualquier maquina: no necesitan webcam ni las fotos de
referencia.
"""

import os
import sys
import tempfile

import cv2
import numpy as np

import VisionArtificial2C2026.TP2_DeteccionYClasificacion.Deteccion.vision as vision


# -----------------------------
# Mini corredor de pruebas
# -----------------------------

PRUEBAS = []


def prueba(fn):
    PRUEBAS.append(fn)
    return fn


# -----------------------------
# Imagenes sinteticas de apoyo
# -----------------------------

def lienzo(fondo=0, ancho=320, alto=240):
    return np.full((alto, ancho), fondo, np.uint8)


def a_color(gris):
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def contorno_de(mascara):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contornos, key=cv2.contourArea)


def ref_moneda(radio=40):
    m = lienzo()
    cv2.circle(m, (160, 120), radio, 255, -1)
    return contorno_de(m)


def ref_lapicera(largo=140, ancho=14):
    m = lienzo()
    x, y = 90, 113
    cv2.rectangle(m, (x, y), (x + largo, y + ancho), 255, -1)
    return contorno_de(m)


def ref_lapicera_rotada(angulo, largo=200, ancho=18):
    m = lienzo()
    caja = cv2.boxPoints(((160, 120), (largo, ancho), angulo))
    cv2.fillPoly(m, [caja.astype(np.int32)], 255)
    return contorno_de(m)


def ref_lentes(radio=28, sep=30, puente=6):
    m = lienzo()
    cv2.circle(m, (160 - sep, 120), radio, 255, -1)
    cv2.circle(m, (160 + sep, 120), radio, 255, -1)
    cv2.rectangle(m, (160 - sep, 120 - puente), (160 + sep, 120 + puente), 255, -1)
    return contorno_de(m)


def escena_hoja_sobre_escritorio():
    """
    Una moneda sobre una hoja blanca que NO llena el encuadre.

    Es la situacion tipica de trabajo: la hoja apoyada sobre un escritorio mas
    oscuro que se ve alrededor.
    """
    img = np.full((480, 640), 85, np.uint8)                  # el escritorio
    cv2.rectangle(img, (120, 60), (520, 420), 238, -1)       # la hoja
    cv2.circle(img, (320, 240), 45, 40, -1)                  # la moneda
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def forma_ajena(puntos):
    m = lienzo()
    cv2.fillPoly(m, [np.array(puntos, np.int32)], 255)
    return contorno_de(m)


def referencias_sinteticas():
    return {
        "moneda": ref_moneda(),
        "lapicera": ref_lapicera(),
        "lentes": ref_lentes(),
    }


def umbrales(valor=0.5):
    return {"moneda": valor, "lapicera": valor, "lentes": valor}


# -----------------------------
# Escala de grises
# -----------------------------

@prueba
def test_a_gris_devuelve_una_sola_banda():
    frame = np.zeros((240, 320, 3), np.uint8)

    gris = vision.a_gris(frame)

    assert gris.shape == (240, 320), f"esperaba (240, 320), obtuve {gris.shape}"
    assert gris.dtype == np.uint8


# -----------------------------
# Binarizacion
# -----------------------------

@prueba
def test_binarizar_manual_deja_blanco_lo_mas_claro_que_el_umbral():
    gris = lienzo(fondo=50)
    cv2.rectangle(gris, (100, 80), (220, 160), 200, -1)

    binaria, umbral_aplicado = vision.binarizar(gris, umbral=127, usar_otsu=False, invertir=False)

    assert umbral_aplicado == 127, f"esperaba 127, obtuve {umbral_aplicado}"
    assert binaria[120, 160] == 255, "el objeto claro tendria que quedar en blanco"
    assert binaria[10, 10] == 0, "el fondo oscuro tendria que quedar en negro"


@prueba
def test_binarizar_invertido_deja_blanco_lo_mas_oscuro_que_el_umbral():
    gris = lienzo(fondo=230)
    cv2.rectangle(gris, (100, 80), (220, 160), 30, -1)

    binaria, _ = vision.binarizar(gris, umbral=127, usar_otsu=False, invertir=True)

    assert binaria[120, 160] == 255, "el objeto oscuro tendria que quedar en blanco"
    assert binaria[10, 10] == 0, "el fondo claro tendria que quedar en negro"


@prueba
def test_otsu_ignora_el_umbral_manual_y_calcula_uno_intermedio():
    # Con ruido, para que el histograma tenga dos modas anchas como en una foto
    # real. Con una imagen de solo dos valores exactos el optimo de Otsu es una
    # meseta y OpenCV devuelve su borde inferior, que no dice nada util.
    rng = np.random.default_rng(0)
    gris = np.clip(rng.normal(50, 12, (240, 320)), 0, 255).astype(np.uint8)
    parche = np.clip(rng.normal(200, 12, (80, 120)), 0, 255).astype(np.uint8)
    gris[80:160, 100:220] = parche

    binaria, umbral_aplicado = vision.binarizar(gris, umbral=5, usar_otsu=True, invertir=False)

    assert 60 < umbral_aplicado < 190, f"Otsu devolvio un umbral raro: {umbral_aplicado}"
    assert binaria[120, 160] == 255
    assert binaria[10, 10] == 0


# -----------------------------
# Morfologia
# -----------------------------

@prueba
def test_limpiar_elimina_el_ruido_sal_y_pimienta_y_conserva_el_objeto():
    binaria = lienzo()
    cv2.rectangle(binaria, (120, 80), (200, 160), 255, -1)
    for x, y in [(20, 20), (60, 200), (300, 30), (280, 210), (40, 120)]:
        binaria[y, x] = 255

    limpia = vision.limpiar(binaria, k=2)

    contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contornos) == 1, f"esperaba 1 objeto tras limpiar, quedaron {len(contornos)}"
    assert cv2.contourArea(contornos[0]) > 4000, "el objeto grande no tendria que desaparecer"


@prueba
def test_limpiar_con_k_cero_no_toca_la_imagen():
    binaria = lienzo()
    binaria[100, 100] = 255

    limpia = vision.limpiar(binaria, k=0)

    assert np.array_equal(limpia, binaria), "con k=0 no se tendria que aplicar morfologia"


# -----------------------------
# Busqueda y filtrado de contornos
# -----------------------------

@prueba
def test_buscar_contornos_descarta_los_de_area_menor_al_minimo():
    binaria = lienzo()
    cv2.circle(binaria, (100, 120), 30, 255, -1)
    cv2.circle(binaria, (260, 60), 3, 255, -1)

    contornos = vision.buscar_contornos(binaria, area_min=500)

    assert len(contornos) == 1, f"esperaba 1 contorno valido, obtuve {len(contornos)}"
    assert cv2.contourArea(contornos[0]) > 2000


@prueba
def test_buscar_contornos_encuentra_un_objeto_metido_dentro_de_un_agujero():
    # Es el caso de la hoja blanca apoyada sobre un escritorio mas oscuro: al
    # binarizar, el escritorio queda como primer plano y la hoja queda como un
    # agujero adentro, con el objeto adentro del agujero. RETR_EXTERNAL devuelve
    # solo los contornos mas externos e ignora todo lo anidado, asi que perdia
    # el objeto.
    binaria = lienzo(fondo=255, ancho=640, alto=480)
    cv2.rectangle(binaria, (120, 60), (520, 420), 0, -1)   # la hoja
    cv2.circle(binaria, (320, 240), 45, 255, -1)           # el objeto encima

    contornos = vision.buscar_contornos(binaria, area_min=500)

    assert len(contornos) == 1, f"esperaba encontrar solo el objeto, obtuve {len(contornos)}"
    area = cv2.contourArea(contornos[0])
    assert 5000 < area < 7500, f"area inesperada para el objeto: {area}"


@prueba
def test_buscar_contornos_no_confunde_el_agujero_con_un_objeto():
    # El borde de la hoja no es un objeto: es el agujero del fondo. No tiene que
    # aparecer como candidato aunque entre en el filtro de area.
    binaria = lienzo(fondo=255, ancho=640, alto=480)
    cv2.rectangle(binaria, (200, 150), (440, 330), 0, -1)

    contornos = vision.buscar_contornos(binaria, area_min=500)

    assert contornos == [], f"el agujero no es un objeto, pero devolvio {len(contornos)}"


@prueba
def test_buscar_contornos_descarta_el_contorno_que_ocupa_casi_todo_el_cuadro():
    binaria = lienzo(fondo=255)

    contornos = vision.buscar_contornos(binaria, area_min=10)

    assert contornos == [], "un contorno que ocupa todo el cuadro es fondo mal binarizado"


# -----------------------------
# Clasificacion con matchShapes
# -----------------------------

@prueba
def test_un_circulo_se_clasifica_como_moneda():
    m = lienzo()
    cv2.circle(m, (160, 120), 55, 255, -1)
    candidato = contorno_de(m)

    nombre, distancia = vision.clasificar(candidato, referencias_sinteticas(), umbrales())

    assert nombre == "moneda", f"esperaba moneda, obtuve {nombre} (distancia {distancia})"


@prueba
def test_una_barra_alargada_se_parece_mas_a_la_lapicera_que_al_resto():
    # Sin umbrales: lo que se prueba es cual referencia queda MAS CERCA, que es
    # una propiedad de la forma y no depende de la escala de la metrica. Un
    # rectangulo fino perfecto es un caso degenerado para los momentos de Hu de
    # orden alto (varios se van casi a cero), asi que su distancia absoluta no
    # dice mucho; lo que si tiene que cumplirse es que gane la clase correcta.
    m = lienzo()
    cv2.rectangle(m, (60, 110), (260, 128), 255, -1)
    candidato = contorno_de(m)

    nombre, _ = vision.clasificar(candidato, referencias_sinteticas(), umbrales(valor=float("inf")))

    assert nombre == "lapicera", f"la mas cercana tendria que ser lapicera, fue {nombre}"


@prueba
def test_una_barra_rotada_sigue_estando_mas_cerca_de_la_lapicera():
    # Los momentos de Hu son invariantes a la rotacion en el plano, asi que girar
    # la barra no puede cambiar cual referencia gana. Con la metrica I1 esto
    # fallaba: una barra rotada 30 grados se clasificaba como moneda.
    candidato = ref_lapicera_rotada(30)

    nombre, _ = vision.clasificar(candidato, referencias_sinteticas(), umbrales(valor=float("inf")))

    assert nombre == "lapicera", f"la mas cercana tendria que ser lapicera, fue {nombre}"


@prueba
def test_sin_candidatos_dentro_del_umbral_la_forma_es_desconocida():
    m = lienzo()
    cv2.circle(m, (160, 120), 55, 255, -1)
    candidato = contorno_de(m)

    nombre, _ = vision.clasificar(candidato, referencias_sinteticas(), umbrales(valor=1e-9))

    assert nombre == vision.DESCONOCIDO, "con umbral casi cero todo tendria que ser desconocido"


@prueba
def test_cada_objeto_sigue_siendo_el_mas_cercano_a_su_clase_cambiando_tamano_y_giro():
    # Cambiar el tamano o girar el objeto en el plano no puede cambiar cual
    # referencia gana: los momentos de Hu son invariantes a las dos cosas.
    referencias = referencias_sinteticas()
    casos = [
        ("moneda", ref_moneda(radio=58)),
        ("moneda", ref_moneda(radio=25)),
        ("lapicera", ref_lapicera(largo=180, ancho=18)),
        ("lapicera", ref_lapicera_rotada(30)),
        ("lapicera", ref_lapicera_rotada(75)),
        ("lentes", ref_lentes(radio=38, sep=41, puente=8)),
        ("lentes", ref_lentes(radio=20, sep=22, puente=4)),
    ]

    for esperado, candidato in casos:
        obtenido, _ = vision.clasificar(candidato, referencias, umbrales(valor=float("inf")))

        assert obtenido == esperado, f"esperaba {esperado}, la mas cercana fue {obtenido}"


# Envolvente medida sobre las fotos de referencia reales con la metrica I3: la
# peor distancia del objeto contra su propia referencia a lo largo de cambios de
# tamano, giros en el plano e inclinaciones, y la distancia de la forma ajena
# mas cercana. El detalle esta en el comentario de UMBRALES_DEFECTO, en
# vision.py. Estas dos tablas son las que justifican los umbrales que se
# entregan, y esta prueba existe para que nadie los toque sin volver a medir.
PEOR_DISTANCIA_MEDIDA = {"lentes": 0.128, "lapicera": 0.695, "moneda": 0.171}
FORMA_AJENA_MAS_CERCANA = {"lentes": 0.372, "lapicera": 1.110, "moneda": 0.200}


@prueba
def test_los_umbrales_por_defecto_cubren_la_variacion_real_medida():
    # Esta es la prueba del problema que rompia el sistema en uso real. Los
    # umbrales originales se calibraron con figuras ideales comparadas contra si
    # mismas en la misma pose, y en vivo eso no pasa nunca: el objeto se ve mas
    # chico y apoyado distinto, la distancia sube, y todo salia DESCONOCIDO.
    for nombre in vision.NOMBRES:
        umbral = vision.UMBRALES_DEFECTO[nombre]

        assert umbral >= PEOR_DISTANCIA_MEDIDA[nombre], (
            f"el umbral de {nombre} ({umbral}) es mas chico que la peor distancia "
            f"medida del objeto real ({PEOR_DISTANCIA_MEDIDA[nombre]}): en vivo va "
            f"a salir DESCONOCIDO"
        )
        assert umbral < FORMA_AJENA_MAS_CERCANA[nombre], (
            f"el umbral de {nombre} ({umbral}) llega hasta la forma ajena mas "
            f"cercana ({FORMA_AJENA_MAS_CERCANA[nombre]}): va a dar falsos positivos"
        )


@prueba
def test_clasificar_sin_referencias_devuelve_desconocido():
    m = lienzo()
    cv2.circle(m, (160, 120), 55, 255, -1)
    candidato = contorno_de(m)

    nombre, distancia = vision.clasificar(candidato, {}, {})

    assert nombre == vision.DESCONOCIDO
    assert distancia is None


# -----------------------------
# Polaridad automatica de las fotos de referencia
# -----------------------------

@prueba
def test_fondo_es_claro_detecta_bien_las_dos_polaridades():
    claro = lienzo(fondo=235)
    cv2.circle(claro, (160, 120), 50, 20, -1)
    oscuro = lienzo(fondo=20)
    cv2.circle(oscuro, (160, 120), 50, 235, -1)

    assert vision.fondo_es_claro(claro) is True
    assert vision.fondo_es_claro(oscuro) is False


@prueba
def test_fondo_es_claro_funciona_con_una_foto_subexpuesta():
    # Una hoja blanca fotografiada con poca luz sale gris (~100), no blanca. En
    # valor absoluto queda por debajo de 127, pero sigue siendo el fondo claro:
    # lo que importa es si el borde es mas claro que el objeto, no cuanto vale.
    gris = lienzo(fondo=100)
    cv2.circle(gris, (160, 120), 50, 35, -1)

    assert vision.fondo_es_claro(gris) is True


@prueba
def test_fondo_es_claro_funciona_con_un_objeto_claro_sobre_fondo_medio():
    # El caso simetrico: un objeto muy claro sobre un fondo gris medio. El fondo
    # esta por encima de 127 pero es el oscuro de los dos.
    gris = lienzo(fondo=150)
    cv2.circle(gris, (160, 120), 50, 245, -1)

    assert vision.fondo_es_claro(gris) is False


@prueba
def test_contorno_principal_con_una_foto_subexpuesta_encuentra_el_objeto():
    # La foto de referencia sacada con poca luz tiene que dar el contorno del
    # objeto, no el del cuadro entero.
    gris = lienzo(fondo=100, ancho=640, alto=480)
    cv2.circle(gris, (320, 240), 60, 35, -1)

    contorno = vision.contorno_principal(a_color(gris))

    assert contorno is not None, "tendria que encontrar el objeto"
    area = cv2.contourArea(contorno)
    assert 9000 < area < 12000, f"esperaba el circulo (~10800), obtuve area {area}"


@prueba
def test_contorno_principal_prefiere_el_objeto_a_una_sombra_pegada_al_borde():
    # Una sombra, o el borde del escritorio asomando en una esquina, puede ser
    # mas grande que el objeto y ganarle por area. El objeto de una foto de
    # referencia no toca los bordes (es la misma suposicion que hace
    # fondo_es_claro), asi que se prefiere el contorno despegado del borde.
    gris = lienzo(fondo=100, ancho=640, alto=480)
    cv2.circle(gris, (480, 150), 45, 30, -1)              # el objeto
    cv2.rectangle(gris, (0, 300), (220, 480), 30, -1)     # la sombra, mas grande

    contorno = vision.contorno_principal(a_color(gris))

    x, y, w, h = cv2.boundingRect(contorno)
    assert x > 0 and y > 0, f"eligio la sombra del borde: caja=({x},{y},{w},{h})"
    area = cv2.contourArea(contorno)
    assert 5000 < area < 7500, f"esperaba el circulo (~6100), obtuve {area}"


@prueba
def test_contorno_principal_igual_sirve_si_el_objeto_llega_al_borde():
    # Si ningun contorno esta despegado del borde, se usa el mas grande igual:
    # una lapicera larga puede cruzar todo el cuadro.
    gris = lienzo(fondo=100, ancho=640, alto=480)
    cv2.rectangle(gris, (0, 200), (560, 260), 30, -1)

    contorno = vision.contorno_principal(a_color(gris))

    assert contorno is not None, "no tendria que descartar el objeto por tocar el borde"
    assert cv2.contourArea(contorno) > 25000, "esperaba la barra completa"


@prueba
def test_contorno_principal_rechaza_una_referencia_que_ocupa_todo_el_cuadro():
    # Si lo unico que se detecta es practicamente el cuadro entero, la foto no
    # sirve como referencia. Mejor devolver None y que main.py lo avise, antes
    # que cargar una referencia inservible en silencio.
    gris = lienzo(fondo=30, ancho=640, alto=480)
    cv2.rectangle(gris, (10, 10), (630, 470), 200, -1)

    assert vision.contorno_principal(a_color(gris)) is None


@prueba
def test_contorno_principal_encuentra_el_objeto_sobre_fondo_claro():
    gris = lienzo(fondo=235)
    cv2.circle(gris, (160, 120), 50, 20, -1)

    contorno = vision.contorno_principal(a_color(gris))

    assert contorno is not None, "tendria que encontrar el objeto oscuro sobre fondo claro"
    area = cv2.contourArea(contorno)
    assert 6000 < area < 9500, f"area inesperada para un circulo de radio 50: {area}"


@prueba
def test_contorno_principal_encuentra_el_objeto_sobre_fondo_oscuro():
    gris = lienzo(fondo=20)
    cv2.circle(gris, (160, 120), 50, 235, -1)

    contorno = vision.contorno_principal(a_color(gris))

    assert contorno is not None, "tendria que encontrar el objeto claro sobre fondo oscuro"
    area = cv2.contourArea(contorno)
    assert 6000 < area < 9500, f"area inesperada para un circulo de radio 50: {area}"


@prueba
def test_cargar_referencias_lee_las_fotos_y_avisa_cuales_faltan():
    carpeta = tempfile.mkdtemp()
    gris = lienzo(fondo=235)
    cv2.circle(gris, (160, 120), 50, 20, -1)
    cv2.imwrite(os.path.join(carpeta, "moneda.jpg"), a_color(gris))

    referencias, faltantes, avisos = vision.cargar_referencias(carpeta)

    assert list(referencias) == ["moneda"], f"esperaba solo moneda, obtuve {list(referencias)}"
    assert sorted(faltantes) == ["lapicera", "lentes"], f"faltantes mal reportados: {faltantes}"
    assert avisos == [], f"la foto es buena, no tendria que avisar nada: {avisos}"
    assert cv2.contourArea(referencias["moneda"]) > 5000


@prueba
def test_cargar_referencias_en_carpeta_inexistente_no_rompe():
    referencias, faltantes, avisos = vision.cargar_referencias("carpeta_que_no_existe")

    assert referencias == {}
    assert sorted(faltantes) == sorted(vision.NOMBRES)
    assert avisos == []


@prueba
def test_cargar_referencias_avisa_si_el_objeto_queda_cortado_por_el_borde():
    # Si el objeto toca el borde de la foto, la suposicion de fondo_es_claro se
    # rompe y ademas no se puede distinguir el objeto de una sombra pegada al
    # borde. Se carga igual, pero avisando, para que no falle en silencio.
    carpeta = tempfile.mkdtemp()
    gris = lienzo(fondo=100, ancho=640, alto=480)
    cv2.circle(gris, (610, 240), 60, 30, -1)
    cv2.imwrite(os.path.join(carpeta, "moneda.jpg"), a_color(gris))

    referencias, _, avisos = vision.cargar_referencias(carpeta)

    assert "moneda" in referencias, "se tiene que cargar igual, solo avisando"
    assert any("moneda" in aviso for aviso in avisos), f"tendria que avisar de moneda: {avisos}"


# -----------------------------
# Pipeline completo
# -----------------------------

@prueba
def test_procesar_detecta_y_clasifica_una_moneda_en_el_frame():
    gris = lienzo(fondo=235)
    cv2.circle(gris, (160, 120), 45, 25, -1)
    frame = a_color(gris)
    params = vision.Parametros(
        umbral=127, usar_otsu=True, invertir=True, kernel=2,
        area_min=800, umbrales=umbrales(),
    )

    resultado = vision.procesar(frame, params, referencias_sinteticas())

    assert len(resultado.detecciones) == 1, f"esperaba 1 objeto, obtuve {len(resultado.detecciones)}"
    deteccion = resultado.detecciones[0]
    assert deteccion.nombre == "moneda", f"esperaba moneda, obtuve {deteccion.nombre}"
    _, _, w, h = deteccion.rect
    assert 60 < w < 110 and 60 < h < 110, f"rectangulo inesperado: {deteccion.rect}"


@prueba
def test_procesar_detecta_un_objeto_sobre_una_hoja_que_no_llena_el_encuadre():
    # Con la hoja apoyada sobre un escritorio mas oscuro, Otsu separa escritorio
    # de hoja, no objeto de hoja: el escritorio queda como primer plano y la
    # moneda queda anidada dentro del agujero de la hoja.
    frame = escena_hoja_sobre_escritorio()

    resultado = vision.procesar(frame, vision.Parametros(), referencias_sinteticas())

    assert len(resultado.detecciones) == 1, (
        f"esperaba detectar la moneda sobre la hoja, obtuve {len(resultado.detecciones)}"
    )
    deteccion = resultado.detecciones[0]
    x, y, w, h = deteccion.rect
    assert 260 < x < 300 and 70 < w < 110, f"rectangulo inesperado: {deteccion.rect}"
    assert deteccion.nombre == "moneda", f"esperaba moneda, obtuve {deteccion.nombre}"


@prueba
def test_procesar_marca_como_desconocido_lo_que_no_se_parece_a_nada():
    gris = lienzo(fondo=235)
    puntos = np.array([[80, 60], [240, 90], [140, 130], [250, 190], [70, 170]], np.int32)
    cv2.fillPoly(gris, [puntos], 25)
    frame = a_color(gris)
    params = vision.Parametros(
        umbral=127, usar_otsu=True, invertir=True, kernel=2,
        area_min=800, umbrales=umbrales(valor=0.05),
    )

    resultado = vision.procesar(frame, params, referencias_sinteticas())

    assert len(resultado.detecciones) == 1
    assert resultado.detecciones[0].nombre == vision.DESCONOCIDO


@prueba
def test_procesar_expone_los_pasos_intermedios():
    frame = a_color(lienzo(fondo=235))
    params = vision.Parametros(
        umbral=127, usar_otsu=False, invertir=True, kernel=2,
        area_min=800, umbrales=umbrales(),
    )

    resultado = vision.procesar(frame, params, {})

    assert resultado.gris.shape == (240, 320), "tendria que devolver la imagen en grises"
    assert resultado.binaria.shape == (240, 320), "tendria que devolver la imagen binaria"
    assert resultado.umbral_aplicado == 127


# -----------------------------
# Anotacion
# -----------------------------

@prueba
def test_anotar_no_modifica_el_frame_original():
    gris = lienzo(fondo=235)
    cv2.circle(gris, (160, 120), 45, 25, -1)
    frame = a_color(gris)
    copia = frame.copy()
    params = vision.Parametros(
        umbral=127, usar_otsu=True, invertir=True, kernel=2,
        area_min=800, umbrales=umbrales(),
    )

    resultado = vision.procesar(frame, params, referencias_sinteticas())
    anotado = vision.anotar(frame, resultado.detecciones)

    assert np.array_equal(frame, copia), "anotar() tendria que trabajar sobre una copia"
    assert anotado.shape == frame.shape
    assert not np.array_equal(anotado, frame), "anotar() tendria que dibujar algo"


@prueba
def test_cada_clase_conocida_tiene_un_color_asignado():
    for nombre in vision.NOMBRES:
        assert nombre in vision.COLORES, f"falta el color de {nombre}"
    assert vision.COLORES[vision.DESCONOCIDO] == (0, 0, 255), "los desconocidos van en rojo"


# -----------------------------
# Coherencia entre vision.py y main.py
# -----------------------------

@prueba
def test_la_barra_de_info_no_tapa_nada_de_la_imagen_anotada():
    # El panel de datos va en una franja debajo, no encima: si se dibujara
    # arriba de la imagen taparia la etiqueta de cualquier objeto que caiga en
    # esa esquina.
    import VisionArtificial2C2026.TP2_DeteccionYClasificacion.Deteccion.main as main

    anotado = np.full((240, 320, 3), 90, np.uint8)

    vista = main.componer_vista(anotado, ["umbral: 131", "objetos: 3"], aviso="faltan referencias")

    assert vista.shape[1] == anotado.shape[1], "la franja tiene que ser del ancho de la imagen"
    assert vista.shape[0] > anotado.shape[0], "la franja tiene que agregar alto"
    assert np.array_equal(vista[:240], anotado), "la imagen anotada tiene que quedar intacta"


@prueba
def test_reconoce_un_frame_negro_de_camara():
    # Si otra aplicacion tiene tomada la webcam, Windows deja abrir el
    # dispositivo pero entrega frames negros. Sin este chequeo el programa
    # mostraba negro sin decir nada y no habia forma de saber que pasaba.
    assert vision.parece_negra(np.zeros((480, 640, 3), np.uint8)) is True
    assert vision.parece_negra(np.full((480, 640, 3), 130, np.uint8)) is False


@prueba
def test_una_habitacion_oscura_no_cuenta_como_camara_bloqueada():
    # Poca luz no es lo mismo que la camara bloqueada: si el aviso saltara con
    # cualquier imagen oscura seria ruido y se aprenderia a ignorarlo.
    apenas_visible = np.full((480, 640, 3), 25, np.uint8)

    assert vision.parece_negra(apenas_visible) is False


@prueba
def test_las_barras_arrancan_en_los_umbrales_por_defecto_de_vision():
    # Un solo lugar define los umbrales. Si se cambian en vision.py, las barras
    # tienen que arrancar ahi mismo.
    import VisionArtificial2C2026.TP2_DeteccionYClasificacion.Deteccion.main as main

    iniciales = {etiqueta: inicial for etiqueta, _, inicial in main.TRACKBARS}

    for nombre in vision.NOMBRES:
        etiqueta = f"Dist {nombre} x100"
        assert etiqueta in iniciales, f"falta la barra {etiqueta}"
        esperado = round(vision.UMBRALES_DEFECTO[nombre] * 100)
        assert iniciales[etiqueta] == esperado, (
            f"{etiqueta} arranca en {iniciales[etiqueta]} pero vision.py dice {esperado}"
        )

    assert iniciales["Kernel morf"] == vision.KERNEL_DEFECTO, (
        f"la barra de kernel arranca en {iniciales['Kernel morf']} pero vision.py "
        f"dice {vision.KERNEL_DEFECTO}"
    )


@prueba
def test_la_referencia_se_extrae_con_el_mismo_kernel_que_usa_la_deteccion():
    # Si la referencia se limpia con un elemento estructural distinto al de la
    # deteccion, las dos siluetas quedan deformadas de forma distinta y la
    # distancia sube sin motivo. Antes contorno_principal tenia un k=3 fijo
    # mientras la deteccion arrancaba con el valor de la barra.
    assert vision.Parametros().kernel == vision.KERNEL_DEFECTO


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    fallos = 0

    for fn in PRUEBAS:
        try:
            fn()
        except AssertionError as e:
            fallos += 1
            print(f"FALLA  {fn.__name__}\n         {e}")
        except Exception as e:
            fallos += 1
            print(f"ERROR  {fn.__name__}\n         {type(e).__name__}: {e}")
        else:
            print(f"ok     {fn.__name__}")

    print()
    print(f"{len(PRUEBAS) - fallos}/{len(PRUEBAS)} pruebas pasaron")
    sys.exit(1 if fallos else 0)
