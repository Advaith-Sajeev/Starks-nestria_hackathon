from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import userdata

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
