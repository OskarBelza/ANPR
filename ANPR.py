import cv2
import numpy as np
import imutils
from skimage.segmentation import clear_border
import pytesseract
import xml.etree.ElementTree as ET

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ANPR:

    def __init__(self):
        pass

    @staticmethod
    def imshow(image, name):
        cv2.imshow(f'{name}', image)
        cv2.waitKey(0)

    @staticmethod
    def countours1(contours):
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                return approx

    @staticmethod
    def countours2(contours, image):
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            ar = w / float(h)

            if ar >= 4 and ar <= 5:
                licensePlate = image[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                return contour

    @staticmethod
    def annotations(img_annotation):
        tree = ET.parse(img_annotation)
        root = tree.getroot()

        license_obj = root.find('object')
        if license_obj is not None:
            bndbox = license_obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            return (xmin, ymin, xmax, ymax)
        else:
            print('No license plate in the annotation file')
            return None

    @staticmethod
    def intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    @staticmethod
    def OCR(image):
        config = '--oem 3 --psm 7'
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()

    @staticmethod
    def anpr(img_path, img_annotation):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        gray = cv2.GaussianBlur(gray, (3, 3), 1)

        edged = cv2.Canny(gray, 30, 200)
        edged = clear_border(edged)

        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = ANPR.countours1(contours)

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)  # Rysowanie białych wypełnionych konturów
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (y, x) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        boxA = ANPR.annotations(img_annotation)
        boxB = (x1, y1, x2, y2)

        print(f'IOU: {ANPR.intersection_over_union(boxA, boxB):2f}')

        license_plate_image = img[y1:y2, x1:x2]

        plate_text = ANPR.OCR(license_plate_image)
        print(f"Rozpoznana tablica rejestracyjna: {plate_text}")

        ANPR.imshow(gray, 'gray')
        ANPR.imshow(edged, 'edges')
        ANPR.imshow(mask, 'mask')
        ANPR.imshow(new_image, 'new image')
        ANPR.imshow(license_plate_image, 'License Plate')
