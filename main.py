from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io
import asyncio
import gc
import uuid

# Initialize the FastAPI app
app = FastAPI()

# Initialize the OCR model at startup
ocr = None

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust this to specific domains if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define a semaphore to limit concurrent OCR processing
semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent requests

# Set a maximum file size limit for uploaded images (e.g., 5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Queue to hold incoming requests and track active SSE connections
request_queue = []
clients = {}
active_sse_connections = set()  # Track active SSE connections

@app.on_event("startup")
async def load_ocr_model():
    global ocr
    # Disable angle classification to save memory
    ocr = PaddleOCR(use_angle_cls=False, lang='en')  # Load the model during startup

# Helper function to perform OCR on the image and return bounding boxes and texts
def ocr_process(image):
    img_array = np.array(image.convert("L"))  # Convert to grayscale to reduce memory usage
    result = ocr.ocr(img_array)
    if result == [None]:
        return []

    box_data = []
    for elements in result[0]:
        box, (text, confidence) = elements[0], elements[1]
        if confidence > 0.1:  # Filter based on confidence
            x_coordinates, y_coordinates = [pt[0] for pt in box], [pt[1] for pt in box]
            left, top = min(x_coordinates), min(y_coordinates)
            width, height = max(x_coordinates) - left, max(y_coordinates) - top
            box_data.append({'left': left, 'top': top, 'width': width, 'height': height, 'text': text})
    return box_data

# Root endpoint
@app.get("/")
async def root():
    return {"message": "hello v1.0.2 optimization attempt 1"}

# Endpoint to handle image upload
@app.post("/uploadImage")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    request_id = str(uuid.uuid4())
    request_queue.append(request_id)
    clients[request_id] = {"status": "queued", "file": image_data}
    return {"request_id": request_id, "message": "File uploaded successfully. Waiting for processing."}

# SSE endpoint to track queue position
@app.get("/queueStatus")
async def queue_status(request_id: str):
    async def queue_position_update():
        active_sse_connections.add(request_id)
        try:
            while request_id in request_queue:
                position = request_queue.index(request_id)
                yield f"data: {position}\n\n"
                await asyncio.sleep(1)
            yield "data: 0\n\n"  # Notify when it’s their turn
        finally:
            active_sse_connections.discard(request_id)  # Remove on disconnection

    return StreamingResponse(queue_position_update(), media_type="text/event-stream")

# Endpoint to process the image
@app.post("/processImage/{request_id}")
async def process_image(request_id: str):
    async with semaphore:
        if request_id not in clients or clients[request_id]["status"] == "completed":
            raise HTTPException(status_code=404, detail="Request not found or expired.")

        if clients[request_id]["status"] == "processing":
            raise HTTPException(status_code=409, detail="Request is already being processed.")
        
        clients[request_id]["status"] = "processing"
        request_queue.remove(request_id)
        
        image_data = clients[request_id]["file"]
        img = Image.open(io.BytesIO(image_data))
        extracted_boxes = await asyncio.to_thread(ocr_process, img)
        
        clients[request_id]["status"] = "completed"
        clients[request_id]["solution"] = extracted_boxes
        gc.collect()

        return {'solution': extracted_boxes, 'status': 'solved'}

# Background task to clean up disconnected clients
async def cleanup_queue_task():
    while True:
        await asyncio.sleep(30)
        for request_id in list(clients.keys()):
            if (clients[request_id]["status"] == "queued" or clients[request_id]["status"] == "completed") and request_id not in active_sse_connections:
                if request_id in request_queue:
                    request_queue.remove(request_id)
                del clients[request_id]
                print("cleared cache")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_queue_task())
