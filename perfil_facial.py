import cv2
import numpy as np
from datetime import datetime

try:
    # Cargar clasificadores en cascada
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    # Variables para almacenar resultados
    age_estimation = "Analizando..."
    sentiment = "Analizando..."
    race_estimation = "Analizando..."
    gender_estimation = "Analizando..."
    last_face_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crear barra lateral
        height, width = frame.shape[:2]
        sidebar_width = 350
        combined_image = np.zeros((height, width + sidebar_width, 3), dtype=np.uint8)
        combined_image[:, :width] = frame

        # Diseño de la barra lateral
        cv2.rectangle(combined_image, (width, 0), (width + sidebar_width, height), (40, 40, 40), -1)
        cv2.putText(combined_image, "ANÁLISIS FACIAL EN TIEMPO REAL", (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        cv2.line(combined_image, (width, 50), (width + sidebar_width, 50), (100, 100, 100), 1)

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detección de rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))

        if len(faces) > 0:
            last_face_detected = True
            for (x, y, w, h) in faces:
                # Dibujar rectángulo en el rostro
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)

                # Región de interés (ROI)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detección de ojos (para edad y sexo)
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

                # Detección de sonrisa (para emoción)
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=25, minSize=(50, 20))
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)

                # **ESTIMACIÓN DE EDAD (MÁS PRECISA)**
                if len(eyes) >= 2:
                    eye_dist = abs(eyes[0][0] - eyes[1][0])  # Distancia entre ojos
                    face_ratio = w / h
                    
                    if eye_dist > 60 and face_ratio > 0.75:
                        age_estimation = "20-30 años"
                    elif eye_dist > 50 and face_ratio > 0.7:
                        age_estimation = "30-45 años"
                    elif eye_dist > 40:
                        age_estimation = "45-60 años"
                    else:
                        age_estimation = "60+ años"
                else:
                    age_estimation = "No detectable"

                # **DETECCIÓN DE EMOCIÓN (MÁS FINA)**
                if len(smiles) > 0:
                    smile_width = sw
                    if smile_width > 70:
                        sentiment = "😊 Feliz"
                    elif smile_width > 40:
                        sentiment = "🙂 Neutral"
                    else:
                        sentiment = "😐 Serio"
                else:
                    # Si no hay sonrisa, analizar cejas y ojos
                    eyebrow_ratio = (np.mean([eh for (ex, ey, ew, eh) in eyes]) / h) if len(eyes) > 0 else 0
                    if eyebrow_ratio > 0.15:
                        sentiment = "😠 Enfadado"
                    else:
                        sentiment = "😐 Neutral"

                # **ESTIMACIÓN DE ORIGEN RACIAL (MEJORADA)**
                skin_roi = roi_color[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
                skin_color = np.mean(skin_roi, axis=(0, 1))
                
                # Clasificación basada en tonos de piel
                if skin_color[2] > 180 and skin_color[1] > 140 and skin_color[0] > 120:
                    race_estimation = "Europeo/Caucásico"
                elif skin_color[2] > 150 and skin_color[1] > 120 and skin_color[0] < 120:
                    race_estimation = "Asiático Oriental"
                elif skin_color[2] < 120 and skin_color[1] < 100 and skin_color[0] < 100:
                    race_estimation = "Africano/Descendiente"
                elif skin_color[2] > 160 and skin_color[1] > 120 and skin_color[0] > 100:
                    race_estimation = "Latino/Mediterráneo"
                else:
                    race_estimation = "Indio/Medio Oriente"

                # **ESTIMACIÓN DE SEXO (AJUSTADA)**
                jaw_width = w
                eyebrow_thickness = np.mean([eh / 4 for (ex, ey, ew, eh) in eyes]) if len(eyes) > 0 else 0
                
                if jaw_width > 140 and eyebrow_thickness > 6:
                    gender_estimation = "♂ Masculino"
                else:
                    gender_estimation = "♀ Femenino"

        else:
            last_face_detected = False

        # **MOSTRAR RESULTADOS EN BARRA LATERAL**
        y_position = 90
        line_height = 45

        # Edad
        cv2.putText(combined_image, "EDAD:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(combined_image, f"{age_estimation}", (width + 20, y_position + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
        y_position += line_height

        # Emoción
        cv2.putText(combined_image, "EMOCIÓN:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        color = (0, 200, 0) if "Feliz" in sentiment else (0, 120, 255) if "Neutral" in sentiment else (0, 0, 200)
        cv2.putText(combined_image, f"{sentiment}", (width + 20, y_position + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_position += line_height

        # Origen racial
        cv2.putText(combined_image, "ORIGEN ESTIMADO:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(combined_image, f"{race_estimation}", (width + 20, y_position + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        y_position += line_height

        # Sexo
        cv2.putText(combined_image, "SEXO:", (width + 20, y_position), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        gender_color = (255, 0, 255) if "Masculino" in gender_estimation else (255, 100, 180)
        cv2.putText(combined_image, f"{gender_estimation}", (width + 20, y_position + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, gender_color, 2)

        # Estado de detección
        status = "ROSTRO DETECTADO ✅" if last_face_detected else "BUSCANDO ROSTRO..."
        status_color = (0, 255, 0) if last_face_detected else (0, 0, 255)
        cv2.putText(combined_image, status, (width + 20, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Mostrar ventana
        cv2.imshow('Análisis Facial Avanzado', combined_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
