# Deployment Guide

This guide details running the platform locally (Windows/Linux/macOS), running with Docker Compose, and deploying to cloud infrastructure (such as Render, Railway, Fly.io, or AWS ECS).

---

## 1. Local Native Run (Windows PowerShell)

### Step 1: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Verify the Dataset
```powershell
python scripts/verify_dataset.py
```

### Step 4: Train the Model (Optional if artifacts/best_model.pt exists)
```powershell
python scripts/run_training.py --epochs 8
```

### Step 5: Evaluate on Official Test Split
```powershell
python scripts/run_evaluation.py
```

### Step 6: Launch Backend API
```powershell
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Step 7: Launch Streamlit Dashboard (In a second terminal window)
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py --server.port 8501
```
- Dashboard URL: `http://localhost:8501`

---

## 2. Docker & Docker Compose Deployment

The project provides a production-grade multi-container stack separating backend inference from frontend presentation.

### Build and Start Containers
```bash
docker compose up -d --build
```

### Inspect Container Health & Status
```bash
docker compose ps
```

### View Live Service Logs
```bash
# Both services
docker compose logs -f

# Backend API only
docker compose logs -f api

# Frontend Streamlit only
docker compose logs -f frontend
```

### Stopping Containers
```bash
docker compose down
```

---

## 3. Cloud Deployment (Render / Railway / Hugging Face Spaces)

The application is structured for simple cloud deployment:

### Option A: Render / Railway (Two-Service Stack)
1. **Service 1: Backend API**
   - **Environment**: Docker
   - **Build Command**: Uses root `Dockerfile`
   - **Start Command**: `python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000`
   - **Health Check Path**: `/health`
   - **Environment Variables**: `ENVIRONMENT=production`
2. **Service 2: Frontend Dashboard**
   - **Environment**: Docker
   - **Build Command**: Uses root `Dockerfile`
   - **Start Command**: `streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0`
   - **Environment Variables**:
     - `BACKEND_API_URL=https://<your-backend-render-subdomain>.onrender.com`

### Option B: Hugging Face Spaces (Streamlit Space)
1. Set Space SDK to `Streamlit`.
2. Commit repository.
3. The app can run unified or proxy to an internal FastAPI service.
