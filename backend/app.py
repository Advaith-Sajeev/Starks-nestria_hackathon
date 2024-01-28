from fastapi import FastAPI, Body, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from database import userdata

# from main import Detector

app = FastAPI()

# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# sign up
@app.post("/api/signup")
async def sign_up(data: dict):
    name = data.get("name")
    username = data.get("username")
    password = data.get("password")
    count = userdata.count_documents({'username': username})
    if count != 0:
        return False
    userdata.insert_one({'name': name, 'username': username, 'password': password})
    return True


# login
@app.get("/api/login")
async def login(username: str, password: str):
    count = userdata.count_documents({'username': username})
    if count == 0:
        return False
    authData = userdata.find_one({"username": username})
    if authData["password"] == password:
        return True
    return False


# data should be passed in the format given below

# const data = { models: ["m1", "m2", "m3"] };
#
# fetch('http://localhost:8000/models/', {
#   method: 'POST',
#   headers: {
#     'Content-Type': 'application/json',
#   },
#   body: JSON.stringify(data),
# })
@app.post("/models/")
async def get_data(models: List[str] = Body(...), video: UploadFile = File(...)):
    # need to store the video locally and pass it into the detector
    contents = await video.read()
    with open(video.filename, "wb") as f:
        f.write(contents)
    path = video.filename
    detecotor = Detector(path)
    return detecotor.aggregate(models)  # the list of models are passed into aggregate function
