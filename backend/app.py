from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from database import userdata, filedata

# from main.Models

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
@app.get("/signup")
async def sign_up(name: str, username: str, password: str):
    count = userdata.count_documents({'username': username})
    if count != 0:
        raise HTTPException(status_code=400, detail="Username already exists")
    userdata.insert_one({'name': name, 'username': username, 'password': password})
    return {"result": True}


# login
@app.get("/login")
async def login(username: str, password: str):
    count = userdata.count_documents({'username': username})
    if count == 0:
        raise HTTPException(status_code=400, detail="Username already exists")
    authData = userdata.find_one({"username": username})
    if authData["password"] == password:
        return True
    raise HTTPException(status_code=400, detail="Username already exists")


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
async def get_data(username, models: List[str] = Body(...), file: UploadFile = File(...)):
    # need to store the video locally and pass it into the detector
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    path = file.filename
    filedata.insert_one({"username": username, "filename": file.filename})
    # detecotor = Detector(path)
    # return detecotor.aggregate(models)  # the list of models are passed into aggregate function

