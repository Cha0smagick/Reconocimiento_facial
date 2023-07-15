    while True:
        # Leer el fotograma de la cámara
        ret, frame = cap.read()

        # Comprobar si se pudo capturar el fotograma correctamente
        if not ret:
            break

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar los ojos en la imagen
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Dibujar cuadrados vacíos verdes alrededor de los ojos detectados
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar el fotograma resultante
        cv2.imshow('Eye Tracking', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except cv2.error as e:
    print(f"Error OpenCV: {e}")

except IOError as e:
    print(f"I/O Error: {e}")

finally:
    # Liberar los recursos y cerrar las ventanas abiertas
    cap.release()
    cv2.destroyAllWindows()
