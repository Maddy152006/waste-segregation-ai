import cv2
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("model/final_model.keras")

class_names = ['metal', 'paper', 'plastic', 'trash']

# Stability buffer
pred_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 🔥 Define center box
    box_size = 300
    x1 = w//2 - box_size//2
    y1 = h//2 - box_size//2
    x2 = w//2 + box_size//2
    y2 = h//2 + box_size//2

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop center region
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    roi = cv2.resize(roi, (224, 224))
    roi_array = np.array(roi)
    roi_array = preprocess_input(roi_array)
    roi_array = np.expand_dims(roi_array, axis=0)

    # Predict
    prediction = model.predict(roi_array, verbose=0)
    label = class_names[np.argmax(prediction)]

    # Stability filter
    pred_buffer.append(label)
    if len(pred_buffer) == pred_buffer.maxlen:
        final_label = Counter(pred_buffer).most_common(1)[0][0]
    else:
        final_label = label

    # Display label
    cv2.putText(frame, final_label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Center Box Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
