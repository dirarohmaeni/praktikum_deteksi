import cv2
import pytesseract
import matplotlib.pyplot as plt

# Lokasi Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"

def detect_plate_number(image_path):

    # Membaca gambar
    image = cv2.imread(image_path)

    if image is None:
        print("Gambar tidak ditemukan!")
        return

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Cari kontur
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    plate = None

    # Cari bentuk persegi panjang
    for contour in contours:

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.018 * perimeter,
            True
        )

        if len(approx) == 4:
            plate = approx
            break

    if plate is None:
        print("Plat nomor tidak terdeteksi")
        return

    # Gambar kotak plat
    cv2.drawContours(image, [plate], -1, (0, 255, 0), 3)

    x, y, w, h = cv2.boundingRect(plate)

    plate_image = gray[y:y+h, x:x+w]

    # Threshold
    _, thresh = cv2.threshold(
        plate_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Simpan hasil potongan
    cv2.imwrite("plat_hasil.jpg", thresh)

    # OCR
    text = pytesseract.image_to_string(
        thresh,
        config='--psm 8'
    )

    print("Nomor Plat :", text.strip())

    # Tampilkan hasil
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Deteksi Plat")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(thresh, cmap="gray")
    plt.title("Hasil OCR")
    plt.axis("off")

    plt.show()


# Gambar yang digunakan
detect_plate_number("car.jpg")