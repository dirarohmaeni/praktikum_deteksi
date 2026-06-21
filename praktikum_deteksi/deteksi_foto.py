import cv2
import imutils

# Inisialisasi HOG Descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca gambar
image = cv2.imread("orang.jpg")

if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# Resize gambar
image = imutils.resize(image, width=min(800, image.shape[1]))

# Deteksi pejalan kaki
(regions, _) = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.05
)

# Gambar kotak deteksi
for (x, y, w, h) in regions:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

print("Jumlah orang terdeteksi:", len(regions))

cv2.imshow("Deteksi Pejalan Kaki - Foto", image)
cv2.waitKey(0)
cv2.destroyAllWindows()