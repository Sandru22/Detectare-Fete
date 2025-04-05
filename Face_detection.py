import cv2

# Încarcă clasificatorul Haar Cascade pre-antrenat pentru fețe
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Încarcă imaginea
image = cv2.imread('D:\\IS IV s2\\VA\\Proiect\\istockphoto-1344872631-612x612.jpg')


# Converteste imaginea la tonuri de gri (necesar pentru detectare)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectează fețele în imagine
faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,  # Parametru pentru compensarea diferitelor distanțe față de cameră
    minNeighbors=3,   # Parametru pentru calitatea detectării (mai mare = mai puține dar mai sigure detectări)
    minSize=(30, 30)  # Dimensiunea minimă a unei fețe de detectat
)

# Desenează dreptunghiuri în jurul fețelor detectate
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Afișează rezultatul
cv2.imshow('Fete detectate', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvează imaginea cu detectările (opțional)
cv2.imwrite('imagine_cu_fete_detectate.jpg', image)