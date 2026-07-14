import cv2

cascade_path = "haarcascade_frontalcatface.xml"
cat_cascade = cv2.CascadeClassifier(cascade_path)

if cat_cascade.empty():
    print("Error: Could not load haarcascade_frontalcatface.xml")
    exit()

img = cv2.imread("cutecat.jpeg")

if img is None:
    print("Error: Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cats = cat_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=2,
    minSize=(30,30)
)

print("Number of cats detected:", len(cats))

for (x, y, w, h) in cats:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Cat Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()