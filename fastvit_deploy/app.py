import io
import datetime as dt
import os
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import torch
from torchvision import transforms
from PIL import Image

from model import load_model, class_names

app = FastAPI(title="FastViT Lung CT Classifier (Batch + Database)")

model, device = load_model()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

DB_PATH = Path(__file__).with_name("patients.db")
SECRET_KEY = os.getenv("FASTVIT_SECRET_KEY", "FASTVIT_SECRET_KEY_12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("FASTVIT_TOKEN_EXPIRE_MINUTES", "60"))
DEFAULT_ADMIN_USERNAME = os.getenv("FASTVIT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("FASTVIT_ADMIN_PASSWORD", "admin123")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp TEXT
            )
            """
        )


def init_user_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                hashed_password TEXT,
                role TEXT
            )
            """
        )


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = dt.datetime.now(tz=dt.UTC) + dt.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user(username: str):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username, hashed_password, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return row


def ensure_default_admin():
    if not DEFAULT_ADMIN_USERNAME or not DEFAULT_ADMIN_PASSWORD:
        return
    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
        if existing:
            return
        conn.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            (
                DEFAULT_ADMIN_USERNAME,
                get_password_hash(DEFAULT_ADMIN_PASSWORD),
                "ADMIN",
            ),
        )


def save_to_db(patient_name: str, filename: str, prediction: str, confidence: float):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (patient_name, filename, prediction, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                patient_name,
                filename,
                prediction,
                confidence,
                dt.datetime.now(tz=dt.UTC).isoformat(),
            ),
        )


init_db()
init_user_db()
ensure_default_admin()


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        role: str | None = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return {"username": username, "role": role}
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "prediction": class_names[pred],
        "confidence": round(float(confidence) * 100, 2),
    }


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user[1]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token({"sub": user[0], "role": user[2]})
    return {"access_token": access_token, "token_type": "bearer", "role": user[2]}


@app.post("/batch_predict/")
async def batch_predict(
    patient_name: str,
    files: list[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    if current_user["role"] not in {"DOCTOR", "ADMIN"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    results = []

    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = round(float(probs[0][pred].item()) * 100, 2)

        prediction = class_names[pred]
        save_to_db(patient_name, file.filename, prediction, confidence)

        results.append(
            {
                "filename": file.filename,
                "prediction": prediction,
                "confidence": confidence,
            }
        )

    return {
        "patient": patient_name,
        "results": results,
    }


@app.get("/get_records/")
def get_records(current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in {"DOCTOR", "ADMIN"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM predictions").fetchall()

    records = [
        {
            "id": row[0],
            "patient_name": row[1],
            "filename": row[2],
            "prediction": row[3],
            "confidence": row[4],
            "timestamp": row[5],
        }
        for row in rows
    ]

    return {"user": current_user["username"], "role": current_user["role"], "records": records}


@app.post("/create_doctor/")
def create_doctor(
    username: str,
    password: str,
    current_user: dict = Depends(get_current_user),
):
    if current_user["role"] != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only admin can create doctors"
        )

    with sqlite3.connect(DB_PATH) as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
                (username, get_password_hash(password), "DOCTOR"),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            ) from exc

    return {"message": f"Doctor {username} created"}
