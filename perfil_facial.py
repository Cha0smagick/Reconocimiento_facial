import cv2
import numpy as np

try:
    # Cargar los clasificadores en cascada preentrenados
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Inicializar la captura de video
    cap = cv2.VideoCapture(0)

    # Variables para almacenar los resultados
    age_estimation = "Analizando..."
    sentiment = "Analizando..."
    race_estimation = "Analizando..."
    gender_estimation = "Analizando..."
    last_detection_time = 0

    while True:
        # Leer el fotograma de la cámara
        ret, frame = cap.read()

        # Comprobar si se pudo capturar el fotograma correctamente
        if not ret:
            break

        # Crear una imagen ampliada para la barra lateral
        height, width = frame.shape[:2]
        sidebar_width = 300
        combined_image = np.zeros((height, width + sidebar_width, 3), dtype=np.uint8)
        combined_image[:, :width] = frame

        # Dibujar la barra lateral
        cv2.rectangle(combined_image, (width, 0), (width + sidebar_width, height), (50, 50, 50), -1)
        cv2.putText(combined_image, "ANALISIS FACIAL", (width + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Separador
        cv2.line(combined_image, (width, 50), (width + sidebar_width, 50), (255, 255, 255), 1)

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_time = cv2.getTickCount()
        if len(faces) > 0:
            last_detection_time = current_time
            for (x, y, w, h) in faces:
                # Dibujar rectángulo alrededor del rostro
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Región de interés para ojos y boca
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detectar ojos
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Detectar sonrisa/boca
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                    
                    # Análisis basado en características detectadas
                    # 1. Estimación de edad (muy aproximada)
                    eye_size_avg = np.mean([eh for (ex, ey, ew, eh) in eyes]) if len(eyes) > 0 else 0
                    face_ratio = w / h
                    
                    if eye_size_avg > 0:
                        age_estimation = "20-30" if eye_size_avg > 25 else "30-40" if eye_size_avg > 20 else "40+"
                    else:
                        age_estimation = "No detectable"
                    
                    # 2. Estimación de sentimiento
                    smile_width = sw if len(smiles) > 0 else 0
                    sentiment = "Feliz" if smile_width > 50 else "Neutral" if smile_width > 20 else "Serio"
                    
                    # 3. Estimación de origen racial (muy básica)
                    skin_roi = roi_color[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
                    skin_color = np.mean(skin_roi, axis=(0,1))
                    
                    if skin_color[0] > 150 and skin_color[1] > 100 and skin_color[2] > 100:
                        race_estimation = "Europeo/Caucásico"
                    elif skin_color[0] > 100 and skin_color[1] > 120 and skin_color[2] < 100:
                        race_estimation = "Asiático"
                    elif skin_color[0] < 120 and skin_color[1] < 120 and skin_color[2] < 120:
                        race_estimation = "Africano/Descendiente"
                    else:
                        race_estimation = "Latino/Mediterráneo"
                    
                    # 4. Estimación de sexo (basado en rasgos faciales)
                    jaw_width = w
                    eyebrow_thickness = np.mean([eh/5 for (ex, ey, ew, eh) in eyes]) if len(eyes) > 0 else 0
                    
                    gender_estimation = "Masculino" if jaw_width > 150 and eyebrow_thickness > 5 else "Femenino"
        
        # Mostrar información en la barra lateral
        y_position = 90
        line_height = 40
        
        # Edad
        cv2.putText(combined_image, f"Edad estimada:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(combined_image, f"{age_estimation}", (width + 20, y_position + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_position += line_height
        
        # Sentimiento
        cv2.putText(combined_image, f"Estado emocional:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(combined_image, f"{sentiment}", (width + 20, y_position + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if sentiment == "Feliz" else (0, 165, 255) if sentiment == "Neutral" else (0, 0, 255), 2)
        y_position += line_height
        
        # Origen racial
        cv2.putText(combined_image, f"Origen estimado:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(combined_image, f"{race_estimation}", (width + 20, y_position + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_position += line_height
        
        # Sexo
        cv2.putText(combined_image, f"Sexo estimado:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(combined_image, f"{gender_estimation}", (width + 20, y_position + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        y_position += line_height
        
        # Indicador de detección
        detection_status = "ROSTRO DETECTADO" if len(faces) > 0 else "BUSCANDO ROSTRO..."
        status_color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
        cv2.putText(combined_image, detection_status, (width + 20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Mostrar el fotograma resultante con la barra lateral
        cv2.imshow('Analisis Facial Avanzado', combined_image)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except cv2.error as e:
    print(f"Error OpenCV: {e}")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Liberar los recursos y cerrar las ventanas abiertas
    cap.release()
    cv2.destroyAllWindows()
