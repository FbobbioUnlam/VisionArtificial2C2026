import cv2
import numpy as np

# Nombres de las 3 clases de objetos de referencia
nombres_clases = ["Reloj", "Hilo", "Lentes"]
contornos_referencia = []

def nada(x):
    pass

def cargar_referencias():
    archivos = ['./referencias/Reloj.jpg', './referencias/hilo.jpg', './referencias/lentes.jpg']
    
    for i, archivo in enumerate(archivos):
        img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Saltando {archivo} - No se encontró.")
            continue

        # 1. Suavizar la imagen para eliminar ruido o texturas del fondo
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 2. Binarización automática (Otsu). 
        _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Forzar fondo negro: Si la esquina superior izquierda es blanca, invertimos todo
        if binaria[0, 0] == 255:
            binaria = cv2.bitwise_not(binaria)
            
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            # 4. Descartar contornos gigantes (el borde de la foto)
            area_imagen = img.shape[0] * img.shape[1]
            contornos_validos = [c for c in contornos if cv2.contourArea(c) < area_imagen * 0.90]
            
            if contornos_validos:
                # Obtener el objeto real más grande
                cnt_max = max(contornos_validos, key=cv2.contourArea)
                contornos_referencia.append(cnt_max)
                
                # Ventanas de diagnóstico visual (puedes borrarlas luego)
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_color, [cnt_max], -1, (0, 255, 0), 3)
                # cv2.imshow(f"Referencia OK - {archivo}", img_color)

def main():
    cargar_referencias()
    
    cap = cv2.VideoCapture(1)
    
    # Crear ventana de controles
    cv2.namedWindow('Controles')
    cv2.resizeWindow('Controles', 400, 150)
    
    # Barras de desplazamiento (Trackbars) sugeridas en el proyecto
    cv2.createTrackbar('Umbral', 'Controles', 127, 255, nada)
    cv2.createTrackbar('Tam_Morfologia', 'Controles', 5, 20, nada)
    # OpenCV trackbars usan enteros. Dividiremos este valor por 100 en el código (ej. 15 -> 0.15)
    cv2.createTrackbar('Dist_Match', 'Controles', 20, 200, nada) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Convertir la imagen a monocromática
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplicar un threshold con umbral ajustable
        val_umbral = cv2.getTrackbarPos('Umbral', 'Controles')
        _, binaria = cv2.threshold(gris, val_umbral, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Aplicar operaciones morfológicas para eliminar ruido
        tam_morf = cv2.getTrackbarPos('Tam_Morfologia', 'Controles')
        if tam_morf > 0:
            kernel = np.ones((tam_morf, tam_morf), np.uint8)
            # Opening para quitar ruido externo, Closing para agujeros internos
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
            
        # Mostrar ventana con paso intermedio
        cv2.imshow('Mascara (Paso Intermedio)', binaria)
        
        # 4. Obtener contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        umbral_distancia = cv2.getTrackbarPos('Dist_Match', 'Controles') / 100.0
        
        for cnt in contornos:
            # 5. Filtrar contornos espúreos por área
            if cv2.contourArea(cnt) < 500:
                continue
                
            mejor_match = -1
            menor_distancia = float('inf')
            
            # 6. Comparar cada contorno con los objetos de referencia
            for i, cnt_ref in enumerate(contornos_referencia):
                distancia = cv2.matchShapes(cnt_ref, cnt, cv2.CONTOURS_MATCH_I1, 0)
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_match = i
                    
            # Obtener rectángulo delimitador para la anotación
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 7. Clasificar y generar imagen anotada
            if menor_distancia <= umbral_distancia and mejor_match != -1:
                color = (0, 255, 0) 
                texto = f"{nombres_clases[mejor_match]} ({menor_distancia:.2f})"
            else:
                color = (0, 0, 255) 
                # Ahora veremos QUÉ distancia calculó aunque sea Desconocido
                texto = f"Desconocido (Dist: {menor_distancia:.2f})"
                
            # Dibujar contorno o rectángulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Etiqueta
            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Mostrar output principal
        cv2.imshow('Output - Deteccion', frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()