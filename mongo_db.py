import os
import motor.motor_asyncio
from dotenv import load_dotenv

load_dotenv()

# MongoDB setup
# The user will provide the URI later, so we default to a local instance or None
MONGO_URI = os.getenv("MONGODB_URI")

db = None
client = None

if MONGO_URI:
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client["chronicle"]
        print("MongoDB connection initialized.")
    except Exception as e:
        print(f"Failed to initialize MongoDB: {e}")
        db = None
else:
    print("WARNING: MONGODB_URI not found in environment. Database operations will be skipped or will fail.")

async def get_db():
    return db
