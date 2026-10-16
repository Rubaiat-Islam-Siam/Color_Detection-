
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# HSV range for green
lower_green = np.array([25, 35, 20])
upper_green = np.array([100, 255, 255])

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam")
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Reduce small noise in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of detected regions
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # Ignore tiny regions
        area = cv2.contourArea(contour)

        if area > 500:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Draw rectangle around the object
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Add label
            cv2.putText(
                frame,
                "Green Object",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Green Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
