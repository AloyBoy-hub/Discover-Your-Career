# Job Roadmapper

This repository hosts the backend and AI models for a Job Roadmapper, a system designed to match candidates with jobs using a Two-Tower embedding model and LLM-based reranking. This was done as part of the Techfest 2026 hackathon hosted by NTU SCDS Club.

## Components
- **FastAPI Backend**: `app.py` serves model predictions and parsed resumes.
- **AI Models**:
  - **Stage A**: Two-Tower Model (BERT-based) for retrieving top candidates.
  - **Stage B**: Cross-Encoder / LLM Reranker for fine-grained selection.
- **Frontend**: Located in the `frontend/` directory.

## Setup

### Prerequisites
- Python 3.9+
- Node.js (for frontend)

### Installation
1.  **Backend**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Frontend**:
    ```bash
    cd frontend
    npm install
    ```

## Usage

### Running the API
```bash
uvicorn app:app --reload
```
The API will be available at `http://localhost:8000`.

### Running the Frontend
```bash
cd frontend
npm run dev
```

## Directory Structure
- `app.py`: Main API entry point.
- `backend/`: Core logic for resume parsing and roadmap generation.
- `scripts/`: Offline training and data preparation scripts.
- `artifacts/`: Stores model weights and config.
- `data/`: Dataset storage.