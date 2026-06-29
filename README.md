# GEN-AI Surgery 🔬🤖

### Project Purpose
GEN-AI Surgery is an AI-based system designed for **holistic surgical scene understanding** in robotic and endoscopic procedures. By combining state-of-the-art **transformer-based computer vision models**, the system analyzes surgical video at multiple levels — detecting instruments, generating segmentation masks, recognizing surgical phases and steps, and classifying fine-grained atomic actions in real time.

### Project Applications
- **Real-time surgical assistance** for instrument identification and anatomical guidance during live operations
- **Autonomous & semi-autonomous robotic surgery** with context-aware decision support
- **Surgical training & simulation** with detailed scene annotations and instrument tracking for medical students
- **Post-operative analysis & documentation** for quality control, research, and outcome assessment
- **Workflow analytics** for hospital efficiency and surgical phase benchmarking

---

## Architecture Diagram
*(Add your architecture diagram here)*
`![Architecture Diagram](images/Architecture.png)`

---

## Workflow Explanation

### Inputs:
- **Surgical Video:** Uploaded by the user via the React frontend.
- **Annotated Frames (GraSP Dataset):** Pre-labeled frames with phase, step, instrument, and action annotations in COCO-style JSON format.

### Processing:

**Instrument Segmentation (Mask2Former):**
- Each video frame is paired with a ground-truth mask.
- Mask2Former uses multi-scale and deformable attention to predict pixel-level binary masks for every instrument per frame, even in cluttered or overlapping scenes.

**Instrument Classification (MViT):**
- Individual frames are preprocessed, balanced across the 4 most frequent instrument classes, and fed into MViT.
- The model classifies which surgical tool is present in each frame.

**Atomic Action Detection (MViT):**
- 16-frame video clips are created around key action frames.
- MViT performs multi-label classification across 14 fine-grained atomic actions (cutting, suturing, pulling, suction, etc.) using inverse frequency weighting to handle class imbalance.

**Surgical Phase & Step Recognition (MViT):**
- Video clips are labeled frame-by-frame for both phase and step categories.
- MViT captures both spatial cues and temporal patterns to classify long-term surgical workflow stages and sub-steps simultaneously.

### Outputs:
- **Segmentation masks** for all instruments per frame
- **Instrument type** classification result
- **Current surgical phase and step** label
- **Atomic actions** detected for each instrument
- All results returned as JSON to the React frontend and logged to MongoDB

---

## Technology Stack

| Component                  | Technology / Model Used                      |
|---------------------------|----------------------------------------------|
| Instrument Segmentation   | **Mask2Former (PyTorch)**                    |
| Instrument Classification | **MViT – Multi-Scale Vision Transformer**    |
| Phase & Step Recognition  | **MViT – Multi-Scale Vision Transformer**    |
| Atomic Action Detection   | **MViT – Multi-Scale Vision Transformer**    |
| AI Orchestration          | Python + Flask                               |
| Frontend                  | React.js (Vite + TypeScript)                 |
| UI Components             | ShadCN/UI, TailwindCSS                       |
| API State Management      | TanStack React Query                         |
| Backend API               | Flask + PyTorch + OpenCV                     |
| Database                  | MongoDB (inference logs & session metadata)  |
| Authentication            | JWT                                          |
| Dataset                   | **GraSP** (13 surgeries, ~32h video)         |

---

## Reference Models & Research

### 1. Mask2Former — Instrument Segmentation
- A transformer-based model for panoptic, instance, and semantic segmentation.
- Uses **multi-scale attention** and **deformable attention** to capture both global context and fine-grained instrument edges.
- Trained on 3,449 labeled frames (2,324 train / 1,125 test) with per-instrument binary masks.

✅ **Strengths:**
- Handles multiple overlapping instruments simultaneously.
- Achieves crisp, accurate masks even in complex surgical scenes.
- Outperforms CNN-based segmentation baselines.

❌ **Limitations:**
- High GPU memory requirements for dense segmentation.
- Performance drops under severe occlusion or extreme lighting variation.

---

### 2. MViT — Multi-Scale Vision Transformer
- A video transformer architecture that processes sequences of frames to understand both spatial and temporal patterns.
- Used for three separate tasks: instrument classification, atomic action detection (16-frame clips), and phase/step recognition.

✅ **Strengths:**
- Captures both short-term motion and long-term procedural context.
- Handles multi-label outputs (multiple simultaneous actions).
- Strong generalization with balanced, augmented training data.

❌ **Limitations:**
- Computationally expensive for long video sequences.
- Sensitive to class imbalance without proper weighting strategies.

---

## Dataset — GraSP

| Property        | Details                                                                             |
|-----------------|-------------------------------------------------------------------------------------|
| **Full Name**   | Holistic and Multi-Granular Surgical Scene Understanding of Prostatectomies         |
| **Scope**       | 13 robotic-assisted prostatectomy surgeries                                         |
| **Duration**    | ~32 hours of surgical video                                                         |
| **Annotations** | 11 phases, 21 steps, 7 instruments, 14 actions                                      |
| **Format**      | COCO-style JSON + frame-level segmentation masks                                    |
| **Sampling**    | 1 frame/second for long-term tasks; key frames every 35 seconds for short-term tasks|
| **Splits**      | Train / Test with cross-validation splits                                           |

### Training Summary

| Task                     | Training Samples            | Test Samples               |
|--------------------------|-----------------------------|----------------------------|
| Instrument Segmentation  | 2,324 images                | 1,125 images               |
| Atomic Action Detection  | 1,506 clips (24,096 frames) | 843 clips (13,488 frames)  |
| Phase & Step Recognition | Per-video clip splits       | Per-video clip splits      |

---

## Getting Started

### Prerequisites

- Node.js ≥ 18
- Python ≥ 3.10
- pip + virtualenv
- MongoDB (local or Atlas URI)
- Docker (optional)
- GPU recommended for model inference

### 1. Clone the Repository

```bash
git clone https://github.com/Praneetha826/GEN_AI_Surgery.git
cd GEN_AI_Surgery
```

### 2. Set Up the Frontend

```bash
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

### 3. Set Up the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Flask API runs at `http://localhost:5000`

### 4. Configure Environment Variables

Create a `.env` file in `/backend`:

```env
MONGO_URI=mongodb://localhost:27017/genai_surgery
JWT_SECRET=your_jwt_secret_here
MODEL_PATH=./models/
```

### 5. Run with Docker (Optional)

```bash
docker-compose up --build
```

---

## How to Run (Manual — All Services)

Run each command in a separate terminal:

```bash
# 1. Start Flask backend (model inference)
cd backend
python app.py
```

```bash
# 2. Start React frontend
npm install
npm run dev
```

---

## API Overview

| Endpoint           | Method | Description                                 |
|--------------------|--------|---------------------------------------------|
| `/api/auth/signup` | POST   | Register a new user                         |
| `/api/auth/login`  | POST   | Login and receive JWT token                 |
| `/api/analyze`     | POST   | Upload surgical video and run full pipeline |
| `/api/results/:id` | GET    | Fetch stored prediction result by ID        |
| `/api/history`     | GET    | Retrieve all past analyses for current user |

**Sample Request:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Authorization: Bearer <your_jwt_token>" \
  -F "video=@/path/to/surgical_video.mp4"
```

**Sample Response:**
```json
{
  "phase": "Dissection",
  "step": "Bladder Neck Dissection",
  "instruments": ["needle_driver", "bipolar_forceps"],
  "atomic_actions": ["cutting", "holding"],
  "segmentation_masks": ["base64_mask_1", "base64_mask_2"],
  "confidence": 0.91
}
```

---

## Challenges Faced

1. **Dataset Quality** — Some frames and annotations were ambiguous or inconsistent, limiting model accuracy.
2. **Class Imbalance** — Rare surgical phases, actions, and instruments were hard to learn; addressed using inverse frequency weighting.
3. **Annotation Mismatches** — Occasional frame-mask-JSON mismatches required careful preprocessing and validation.
4. **Hardware Constraints** — Dense segmentation and video transformer models demanded significant GPU memory; some multi-task goals were simplified to fit available compute.
5. **Real-World Variability** — Differences in lighting, camera angles, and surgeon technique across procedures added complexity.
6. **Generalization** — Risk of overfitting to specific surgeries; mitigated through data augmentation and cross-validation splits.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

> Built by the GEN-AI Surgery Team · B.Tech CSE, KMIT · [Praneetha's LinkedIn](https://linkedin.com/in/praneetha-palakurla)
````
