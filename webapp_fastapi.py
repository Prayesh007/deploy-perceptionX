import os
import io
import cv2
import numpy as np
import subprocess
import tempfile
from bson import ObjectId
from bson.binary import Binary
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from ultralytics import YOLO

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Connect to MongoDB
mongo = AsyncIOMotorClient("mongodb+srv://aitools2104:kDTRxzV6MgO4nicA@cluster0.tqkyb.mongodb.net/?retryWrites=true&w=majority")
db = mongo["test"]
files_collection = db["files"]

# Load YOLO model
weights_path = "yolov11/best.pt"
if not os.path.exists(weights_path):
    raise FileNotFoundError("Model not found at yolov11/best.pt")
model = YOLO(weights_path)
print("✅ YOLO model loaded successfully!")

# Mount templates and static files
templates = Jinja2Templates(directory="views")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Homepage
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("perceps/index.html", {"request": request})

@app.get("/detect", response_class=HTMLResponse)
async def detect(request: Request):
    return templates.TemplateResponse("perceps/detect.html", {"request": request})

# Upload and process file (synchronously)
@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    file_type = "image" if file.content_type.startswith("image") else "video"
    data = await file.read()

    doc = {
        "filename": file.filename,
        "mimetype": file.content_type,
        "data": Binary(data),
        "processedData": None
    }

    result = await files_collection.insert_one(doc)
    file_id = str(result.inserted_id)
    print(f"✅ File saved to DB with ID: {file_id}")

    # Synchronous processing for compatibility on Render
    if file_type == "image":
        await process_image(file_id, data)
    else:
        await process_video(file_id, data)

    return {"fileId": file_id}

# Image processing
async def process_image(file_id, data):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(img)
    results = model(img_np)
    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    await files_collection.update_one(
        {"_id": ObjectId(file_id)},
        {"$set": {"processedData": Binary(buffer.tobytes())}}
    )
    print(f"✅ Processed image saved: {file_id}")

# Video processing
async def process_video(file_id, data):
    input_path = os.path.join(tempfile.gettempdir(), f"{file_id}_input.mp4")
    out_avi = input_path.replace(".mp4", "_out.avi")
    final_mp4 = out_avi.replace(".avi", ".mp4")

    with open(input_path, "wb") as f:
        f.write(data)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_avi, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()

    subprocess.run(["ffmpeg", "-y", "-i", out_avi, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", final_mp4])

    with open(final_mp4, "rb") as f:
        processed = f.read()

    await files_collection.update_one(
        {"_id": ObjectId(file_id)},
        {"$set": {"processedData": Binary(processed), "mimetype": "video/mp4"}}
    )
    print(f"✅ Processed video saved: {file_id}")

# Serve original file
@app.get("/file/{file_id}")
async def get_original(file_id: str):
    doc = await files_collection.find_one({"_id": ObjectId(file_id)})
    if not doc:
        raise HTTPException(404, "File not found")
    return StreamingResponse(io.BytesIO(doc["data"]), media_type=doc["mimetype"])

# Serve processed file
@app.get("/file/{file_id}/processed")
async def get_processed(file_id: str):
    doc = await files_collection.find_one({"_id": ObjectId(file_id)})
    if not doc or not doc.get("processedData"):
        raise HTTPException(404, "Processed file not found")

    mimetype = doc.get("mimetype", "application/octet-stream")
    if mimetype.startswith("video"):
        mimetype = "video/mp4"
    elif mimetype.startswith("image"):
        mimetype = "image/jpeg"

    return StreamingResponse(io.BytesIO(doc["processedData"]), media_type=mimetype)

# Entry point for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp_fastapi:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))




# import os
# import io
# import cv2
# import numpy as np
# import subprocess
# import tempfile
# from bson import ObjectId
# from bson.binary import Binary
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Request
# from fastapi.responses import HTMLResponse, StreamingResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from motor.motor_asyncio import AsyncIOMotorClient
# from ultralytics import YOLO

# app = FastAPI()

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# # MongoDB Atlas
# mongo = AsyncIOMotorClient("mongodb+srv://aitools2104:kDTRxzV6MgO4nicA@cluster0.tqkyb.mongodb.net/?retryWrites=true&w=majority")
# db = mongo["test"]
# files_collection = db["files"]

# # Load YOLO model
# weights_path = "yolov11/best.pt"
# if not os.path.exists(weights_path):
#     raise FileNotFoundError("Model not found at yolov11/best.pt")
# model = YOLO(weights_path)
# print("✅ YOLO model loaded successfully!")

# # Templates and Static
# templates = Jinja2Templates(directory="views")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Home Page
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("perceps/index.html", {"request": request})

# @app.get("/detect", response_class=HTMLResponse)
# async def detect(request: Request):
#     return templates.TemplateResponse("perceps/detect.html", {"request": request})

# # Upload and Process File
# @app.post("/process")
# async def process_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
#     file_type = "image" if file.content_type.startswith("image") else "video"
#     data = await file.read()

#     doc = {
#         "filename": file.filename,
#         "mimetype": file.content_type,
#         "data": Binary(data),
#         "processedData": None
#     }

#     result = await files_collection.insert_one(doc)
#     file_id = str(result.inserted_id)
#     print(f"✅ File saved to DB with ID: {file_id}")

#     background_tasks.add_task(process_file_background, file_id, file_type)
#     return {"fileId": file_id}

# # Background File Processing
# async def process_file_background(file_id, file_type):
#     import asyncio
#     for _ in range(5):  # retry to handle Mongo delay
#         doc = await files_collection.find_one({"_id": ObjectId(file_id)})
#         if doc:
#             break
#         await asyncio.sleep(1)
#     else:
#         print("❌ Document not found after retries.")
#         return

#     if file_type == "image":
#         await process_image(file_id, doc["data"])
#     else:
#         await process_video(file_id, doc["data"])

# async def process_image(file_id, data):
#     image = Image.open(io.BytesIO(data)).convert("RGB")
#     image_np = np.array(image)
#     results = model(image_np)
#     annotated = results[0].plot()

#     _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
#     await files_collection.update_one(
#         {"_id": ObjectId(file_id)},
#         {"$set": {"processedData": Binary(buffer.tobytes())}}
#     )
#     print(f"✅ Processed image saved: {file_id}")

# async def process_video(file_id, data):
#     input_path = os.path.join(tempfile.gettempdir(), "input.mp4")
#     out_avi = input_path.replace(".mp4", "_out.avi")
#     final_mp4 = out_avi.replace(".avi", ".mp4")

#     with open(input_path, "wb") as f:
#         f.write(data)

#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*"XVID")
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(out_avi, fourcc, fps, (width, height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model(frame)
#         annotated = results[0].plot()
#         out.write(annotated)

#     cap.release()
#     out.release()

#     subprocess.run(["ffmpeg", "-y", "-i", out_avi, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", final_mp4])

#     with open(final_mp4, "rb") as f:
#         processed = f.read()

#     await files_collection.update_one(
#         {"_id": ObjectId(file_id)},
#         {"$set": {"processedData": Binary(processed), "mimetype": "video/mp4"}}
#     )
#     print(f"✅ Processed video saved: {file_id}")

# # Serve Original File
# @app.get("/file/{file_id}")
# async def get_original(file_id: str):
#     doc = await files_collection.find_one({"_id": ObjectId(file_id)})
#     if not doc:
#         raise HTTPException(404, "File not found")
#     return StreamingResponse(io.BytesIO(doc["data"]), media_type=doc["mimetype"])

# # Serve Processed File
# @app.get("/file/{file_id}/processed")
# async def get_processed(file_id: str):
#     doc = await files_collection.find_one({"_id": ObjectId(file_id)})
#     if not doc or not doc.get("processedData"):
#         raise HTTPException(404, "Processed file not found")

#     mimetype = doc.get("mimetype", "application/octet-stream")
#     if mimetype.startswith("video"):
#         mimetype = "video/mp4"
#     elif mimetype.startswith("image"):
#         mimetype = "image/jpeg"

#     return StreamingResponse(io.BytesIO(doc["processedData"]), media_type=mimetype)

# # Entry Point for Local Dev (ignored by Render)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("webapp_fastapi:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
