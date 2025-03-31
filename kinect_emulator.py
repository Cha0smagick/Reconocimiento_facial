import cv2
import mediapipe as mp
import numpy as np

class XboxLikeMotionCapture:
    def __init__(self, max_people=4):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True
        )
        self.max_people = max_people  # Máximo número de personas a detectar
    
    def process_frame(self, frame):
        # Convertir BGR a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen y detectar poses
        results = self.pose.process(image)
        
        # Convertir de vuelta a BGR para renderizado
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Dibujar landmarks de la pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        return image, results.pose_landmarks
    
    def get_skeleton_data(self, landmarks):
        if not landmarks:
            return None
            
        # Extraer puntos clave y convertirlos a coordenadas normalizadas
        keypoints = []
        for landmark in landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return np.array(keypoints)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Voltear la imagen para una experiencia tipo espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar el frame
            processed_frame, landmarks = self.process_frame(frame)
            
            # Mostrar el resultado
            cv2.imshow('Xbox-like Motion Capture', processed_frame)
            
            # Obtener datos del esqueleto (para uso posterior)
            skeleton_data = self.get_skeleton_data(landmarks)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Uso del sistema
if __name__ == "__main__":
    motion_capture = XboxLikeMotionCapture(max_people=2)
    motion_capture.run()
