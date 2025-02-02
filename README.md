# Automatic Number Plate Recognition (ANPR)

This project implements **Automatic Number Plate Recognition (ANPR)** using **OpenCV**, **Tesseract OCR**, and image processing techniques.

## Features
- **Detects license plates** from images using contour detection.
- **Extracts text** from detected plates using Tesseract OCR.
- **Compares detected bounding box** with annotated ground truth using Intersection over Union (IoU).
- **Displays processed images** at various stages of detection.

## Installation

### **1. Install Python Dependencies**
Ensure you have Python installed, then install the required packages using:
```sh
pip install opencv-python numpy imutils scikit-image pytesseract
```

### **2. Install Tesseract OCR**
To use OCR functionality, you need to install **Tesseract OCR**.

#### **Windows Installation:**
1. Download the Tesseract installer from:  
   [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and **install all available languages**.
3. Note the installation path, typically:
   ```
   C:\Program Files\Tesseract-OCR
   ```
4. Add this path to your Python script:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
   ```

#### **Linux Installation:**
```sh
sudo apt install tesseract-ocr
```

#### **Mac Installation:**
```sh
brew install tesseract
```

### **3. Verify Tesseract Installation**
Run the following command in the terminal to check if Tesseract is installed correctly:
```sh
tesseract -v
```
If installed successfully, it will display the version number.

## Usage

### **Run the ANPR script**
Call the `anpr` function with an image and its annotation file:
```python
from anpr import ANPR

ANPR.anpr("path_to_image.jpg", "path_to_annotation.xml")
```

### **How It Works?**
1. **Preprocessing**
   - Converts the image to grayscale.
   - Applies Gaussian blur to reduce noise.
   - Detects edges using Canny edge detection.
2. **License Plate Detection**
   - Finds contours and selects the best candidate for a plate.
   - Extracts the detected license plate region.
3. **OCR (Optical Character Recognition)**
   - Reads the text from the extracted license plate using **Tesseract OCR**.
4. **Bounding Box Evaluation (IoU)**
   - Compares the detected bounding box with the ground truth from the XML annotation.

## Example Output
```
IOU: 0.85
Recognized License Plate: ABC1234
```

## File Structure
```
ANPR/
│── anpr.py          # Main ANPR implementation
│── example.jpg      # Sample image for testing
│── annotation.xml   # Ground truth annotation
│── README.md        # Documentation
```

## License
This project is open-source and available under the **MIT License**.

