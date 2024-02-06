from pymongo import MongoClient

connection_str = "mongodb+srv://root:root@cluster0.tnz2ij1.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_str)
usersAuthDB = client["usersAuthDB"]  # Creating a database to store the username and password
userdata = usersAuthDB["userdata"]  # Creating a collection to store the username and password
filedata = usersAuthDB["filedata"]

