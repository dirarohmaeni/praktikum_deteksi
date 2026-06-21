import cv2
import matplotlib.pyplot as plt
import pytesseract

# ==================================================
# KONFIGURASI TESSERACT OCR
# ==================================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"

# ==================================================
# MEMBACA GAMBAR
# ==================================================
image = cv2.imread("car.jpg")

# Menampilkan gambar asli
plt.figure("Gambar Asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# ==================================================
# KONVERSI KE GRAYSCALE
# ==================================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure("Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

# ==================================================
# GAUSSIAN BLUR (MENGHILANGKAN NOISE)
# ==================================================
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

plt.figure("Gaussian Blur")
plt.imshow(blurred, cmap='gray')
plt.axis('off')
plt.show()

# ==================================================
# DETEKSI TEPI MENGGUNAKAN CANNY
# ==================================================
edges = cv2.Canny(blurred, 100, 200)

plt.figure("Deteksi Tepi")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

# ==================================================
# MENCARI KONTUR
# ==================================================
contours, _ = cv2.findContours(
    edges.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Jumlah kontur yang ditemukan:", len(contours))