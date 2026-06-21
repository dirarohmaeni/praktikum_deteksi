import cv2
import imutils

# Inisialisasi HOG Descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca video
cap = cv2.VideoCapture("vid.mp4")

if not cap.isOpened():
    print("Video tidak ditemukan atau rusak!")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(
        frame,
        width=min(800, frame.shape[1])
    )

    # Deteksi manusia
    (regions, _) = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05
    )

    # Kotak deteksi
    for (x, y, w, h) in regions:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    # Tampilkan jumlah orang
    cv2.putText(
        frame,
        f"Jumlah Orang: {len(regions)}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Deteksi Pejalan Kaki - Video",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()