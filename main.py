from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io
import asyncio

# Initialize the FastAPI app
app = FastAPI()

# Initialize the OCR model at startup
ocr = None

@app.on_event("startup")
async def load_ocr_model():
    global ocr
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load the model during startup

# Helper function to perform OCR on the image and return bounding boxes and texts
def ocr_process(image):
    # Convert PIL Image to format expected by OCR
    img_array = np.array(image)
    
    # Perform OCR on the image
    result = ocr.ocr(img_array)
    
    # Ensure OCR result is valid
    if result == [None]:
        return []
    
    # Process the result to extract boxes and texts
    box_data = []
    for elements in result[0]:
        # Each element contains the bounding box and the text with confidence
        box = elements[0]  # [x1, y1], [x2, y2], [x3, y3], [x4, y4]
        text, confidence = elements[1]
        
        if confidence > 0.1:  # Filter based on confidence
            # Calculate bounding box: top, left, width, height
            x_coordinates = [point[0] for point in box]
            y_coordinates = [point[1] for point in box]
            left = min(x_coordinates)
            top = min(y_coordinates)
            width = max(x_coordinates) - left
            height = max(y_coordinates) - top
            
            # Add the box data and text to the list
            box_data.append({
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'text': text
            })
    
    return box_data

# Define the /uploadImage route
@app.post("/uploadImage")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Perform OCR and extract the bounding boxes and text
        extracted_boxes = await asyncio.to_thread(ocr_process, img)
        
        # Return the extracted box data
        return {
            'solution': extracted_boxes,
            'status': 'solved'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
