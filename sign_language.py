import sys
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==============================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================

DATA_DIR = "data_signs"   # Carpeta donde se guardarán datos y modelos
MODEL_FILE = "model.pkl"  # Archivo donde se guardará el modelo entrenado

os.makedirs(DATA_DIR, exist_ok=True)

# Mapeo de índices del modelo -> nombre de la clase (en este ejemplo, letras)
labels_map = {
    0: "A",
    1: "B"
}

# MediaPipe para manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==============================================================
# FUNCIÓN: RECOLECCIÓN DE DATOS
# ==============================================================

def collect_data(label_name="A"):
    """
    Captura frames de la webcam, extrae landmarks de la mano con MediaPipe
    y guarda esos vectores (63 valores: x,y,z por cada uno de los 21 puntos)
    junto con la etiqueta (label_name).
    """
    # Mapeo de la letra (A, B, etc.) a índice numérico (0, 1, etc.)
    label_to_index = {"A": 0, "B": 1}

    if label_name not in label_to_index:
        print(f"Etiqueta '{label_name}' no está soportada en este ejemplo.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[RECOLECCIÓN] No se pudo abrir la webcam.")
        return

    print(f"[RECOLECCIÓN] Presiona 's' para GUARDAR un frame etiquetado como '{label_name}'")
    print(f"[RECOLECCIÓN] Presiona 'q' para SALIR.")

    collected_features = []
    collected_labels = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,              # Procesaremos solo 1 mano principal
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
            # Volver a BGR para dibujar y mostrar
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Si detecta una mano
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibuja la malla de la mano en la imagen
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Collect Data", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Salir
                break
            elif key == ord('s'):
                # Guardar landmarks si hay al menos una mano detectada
                if results.multi_hand_landmarks:
                    # Tomamos la primera mano detectada
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Extraer x, y, z de los 21 puntos
                    data_points = []
                    for lm in hand_landmarks.landmark:
                        data_points.extend([lm.x, lm.y, lm.z])

                    collected_features.append(data_points)
                    collected_labels.append(label_to_index[label_name])
                    print(f"[INFO] Frame guardado para la clase '{label_name}'.")
                else:
                    print("[INFO] No se detectó mano en este frame, no se guardó nada.")

    cap.release()
    cv2.destroyAllWindows()

    # Guardar los datos en .npy
    features_file = os.path.join(DATA_DIR, f"features_{label_name}.npy")
    labels_file = os.path.join(DATA_DIR, f"labels_{label_name}.npy")

    if len(collected_features) > 0:
        np.save(features_file, np.array(collected_features))
        np.save(labels_file, np.array(collected_labels))
        print(f"[RECOLECCIÓN] Guardado: {features_file} y {labels_file}")
    else:
        print("[RECOLECCIÓN] No se guardó nada, no hay datos.")

# ==============================================================
# FUNCIÓN: ENTRENAMIENTO DEL MODELO
# ==============================================================

def train_model():
    """
    Carga todos los .npy de features_*.npy y labels_*.npy (para A, B, etc.),
    entrena un RandomForest y lo serializa en un archivo (model.pkl).
    """
    all_features = []
    all_labels = []

    # Recorremos los archivos en la carpeta data_signs
    for file in os.listdir(DATA_DIR):
        if file.startswith("features_") and file.endswith(".npy"):
            label_name = file.split("_")[1].split(".")[0]  # e.g. "A" o "B"
            features_path = os.path.join(DATA_DIR, file)
            labels_path = os.path.join(DATA_DIR, f"labels_{label_name}.npy")

            if os.path.exists(labels_path):
                X = np.load(features_path)
                y = np.load(labels_path)

                all_features.append(X)
                all_labels.append(y)

    if not all_features:
        print("[TRAIN] No se encontraron datos para entrenar. Recolecta primero.")
        return

    # Concatenar
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"[TRAIN] Forma de X: {X.shape}, Forma de y: {y.shape}")

    # Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Métricas
    y_pred = clf.predict(X_test)
    print("[TRAIN] Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Guardar el modelo en disco con pickle
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    print(f"[TRAIN] Modelo entrenado y guardado en '{MODEL_FILE}'.")

# ==============================================================
# FUNCIÓN: DETECCIÓN EN TIEMPO REAL
# ==============================================================

def detect_signs():
    """
    Carga el modelo desde 'model.pkl', inicia la webcam, extrae landmarks
    de la mano y los pasa al modelo para predecir la clase (A/B), mostrándolo en pantalla.
    """
    # Cargar el modelo desde el archivo
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(f"[DETECT] Modelo cargado desde '{MODEL_FILE}'.")
    except FileNotFoundError:
        print(f"[DETECT] No se encontró '{MODEL_FILE}'. Debes entrenar primero (python sign_language.py train).")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[DETECT] No se pudo abrir la webcam.")
        return

    print("[DETECT] Presiona 'q' para salir.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            sign_text = ""

            if results.multi_hand_landmarks:
                # Tomamos la primer mano detectada
                hand_landmarks = results.multi_hand_landmarks[0]

                # Extraer 21 puntos * (x,y,z) = 63 valores
                data_points = []
                for lm in hand_landmarks.landmark:
                    data_points.extend([lm.x, lm.y, lm.z])

                data_points = np.array(data_points).reshape(1, -1)
                pred = model.predict(data_points)

                # Mapear índice a la letra
                sign_text = labels_map.get(pred[0], "?")

                # Dibujamos landmarks en la imagen
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Mostrar el signo predicho en la esquina superior izquierda
            cv2.putText(frame_bgr, f"Sign: {sign_text}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("Real-Time Detection", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ==============================================================
# MAIN: uso por línea de comandos
# ==============================================================
if __name__ == "__main__":
    """
    Ejemplos de uso:
      1) Recolectar datos para la letra 'A':
         python sign_language.py collect A

      2) Recolectar datos para la letra 'B':
         python sign_language.py collect B

      3) Entrenar el modelo con los datos guardados (A, B, etc.):
         python sign_language.py train

      4) Detección en tiempo real (usa el modelo en 'model.pkl'):
         python sign_language.py detect
    """
    if len(sys.argv) < 2:
        print("Modo de uso:")
        print("  python sign_language.py collect [A|B]")
        print("  python sign_language.py train")
        print("  python sign_language.py detect")
        sys.exit(1)

    command = sys.argv[1]

    if command == "collect":
        if len(sys.argv) == 3:
            label_name = sys.argv[2]
            collect_data(label_name)
        else:
            print("Debes especificar la etiqueta, por ejemplo: python sign_language.py collect A")

    elif command == "train":
        train_model()

    elif command == "detect":
        detect_signs()

    else:
        print(f"Comando '{command}' no reconocido.")