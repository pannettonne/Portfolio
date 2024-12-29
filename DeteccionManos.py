import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=False,       # False para video en vivo
        max_num_hands=2,              # Hasta 2 manos simultáneamente
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir de BGR (OpenCV) a RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Procesar con MediaPipe Hands
            results = hands.process(frame_rgb)

            # Volver a BGR para dibujar y mostrar con OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibuja la malla de la mano en la imagen
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.imshow("Hand Detection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()