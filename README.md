# AI Image Segmentation Project

This project contains a frontend and backend for an AI Image Segmentation web application built with React, Vite, and FastAPI.

## Folder Structure

- `/frontend` - The React application (Vite).
- `/backend` - The Python backend API serving predictions.
- `predict.py`, `train.py`, `test_val.py` - Model training and validation scripts.
- `real_model.pth`, `real_model_v2.pth` - Pre-trained model weights.

## Getting Started

### 1. Start the Backend

The backend uses Python and FastAPI. It relies on PyTorch and other machine learning libraries.

1. Open a new terminal.
2. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
3. Run the fastAPI server:
   ```bash
   python app.py
   ```
   *(The server will typically start on `http://127.0.0.1:5000` or `http://127.0.0.1:8000`, depending on the configuration in `app.py`.)*

### 2. Start the Frontend

The frontend is a modern React application built with Vite.

1. Open a new terminal.
2. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
3. Install dependencies (if you haven't already):
   ```bash
   npm install
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
5. Open the provided `localhost` URL in your browser to view the application.

## Dealing with Git Push Errors

If you encountered a `[rejected]` error when pushing to `main`, it means there are changes perfectly merged in the remote repository that you don't have locally. To integrate them without losing your current code:

```bash
# Pull the latest changes from the remote main branch
git pull origin main --rebase

# Then push your changes again
git push -u origin main
```
