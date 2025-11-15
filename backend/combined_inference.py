# --------------------------------------------------------------
#  combined_inference.py  (FIXED: correct seg + phase/step parsing)
# --------------------------------------------------------------
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os, glob, cv2, torch, jwt as pyjwt
from datetime import datetime
from PIL import Image

# --- Imports from other modules ------------------------------------------------
from instrument_detection import (
    load_model as load_detection_model,
    get_instrument_detection_result,
    DEVICE
)
from instrument_segmentation import (
    load_model as load_segmentation_model,
    get_segmented_image_path,
    UPLOAD_FOLDER as SEG_UPLOAD_FOLDER
)
from atomic_actions import (
    get_atomic_actions_result as run_atomic_actions_inference_core,
    MSVisionTransformer,
    FRAMES_PER_CLIP, NUM_LABELS,
    action_names
)
from phase_step import (
    get_phase_step_result,
    TAPIS_PhaseStep,
    phases_categories,
    steps_categories,
    phase_id2cat, step_id2cat
)
from atomic_actions import init_atomic_actions
from phase_step import init_model as init_phase_step_model
from auth import db, JWT_SECRET_KEY

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'Uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEG_UPLOAD_FOLDER, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
combined_bp = Blueprint('combined_inference', __name__)

# ----------------------------------------------------------------------
#  Model creation + weight loading
# ----------------------------------------------------------------------
print("Creating & loading models...")

phase_name2cat = {item['name']: item for item in phases_categories}
step_name2cat = {item['name']: item for item in steps_categories}

DETECTION_MODEL    = load_detection_model()
SEGMENTATION_MODEL = load_segmentation_model()

ATOMIC_ACTIONS_MODEL = MSVisionTransformer(
    in_ch=3, img_size=112, frames=FRAMES_PER_CLIP, num_classes=NUM_LABELS,
    stage_channels=[64,128,256,512], stage_depths=[2,2,4,2],
    heads=[4,8,8,16], use_checkpoint=False
).to(DEVICE)
ATOMIC_ACTIONS_MODEL = init_atomic_actions(ATOMIC_ACTIONS_MODEL, DEVICE)
print("Atomic actions model loaded")

PHASE_STEP_MODEL = TAPIS_PhaseStep(
    num_phases=len(phase_id2cat), num_steps=len(step_id2cat)
).to(DEVICE)
PHASE_STEP_MODEL = init_phase_step_model(PHASE_STEP_MODEL, DEVICE)
print("Phase/Step model loaded")

# ----------------------------------------------------------------------
#  Helper wrappers
# ----------------------------------------------------------------------
def run_atomic_actions_inference(vid_path, model):
    return run_atomic_actions_inference_core(vid_path, model, DEVICE)

def run_phase_step_inference(vid_path, model):
    return get_phase_step_result(vid_path, model, DEVICE)

# ----------------------------------------------------------------------
#  Generate full-duration annotated video (FIXED)
# ----------------------------------------------------------------------
def generate_annotated_video(input_video_path, output_path,
                            phase, step, actions,
                            detection_model, segmentation_model):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open input video")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    line = 35

    # ----- AI every ~1 sec (≈1 FPS) -----
    sample_every_n = 3        # ~1 FPS
    last_ai_frame  = -sample_every_n

    last_probs   = {}
    last_seg_png = None          # full path to the *last* saved PNG
    used_pngs    = set()

    print(f"Processing {total} frames @ {fps:.2f} FPS (AI every {sample_every_n} frames)")

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        annotated = frame_bgr.copy()

        # ---------- Run AI ----------
        if frame_idx - last_ai_frame >= sample_every_n:
            pil_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            # 1. Detection
            _, probs = get_instrument_detection_result(pil_rgb, detection_model, DEVICE)
            last_probs = {k: v for k, v in probs.items() if v > 0.2}

            # 2. Segmentation – **save PNG in the SAME folder as the video**
            seg_name = f"seg_{frame_idx:06d}.png"
            # The helper returns a *relative* path; we force it into UPLOAD_FOLDER
            _ = get_segmented_image_path(frame_bgr, segmentation_model, DEVICE, seg_name)
            seg_path = os.path.join(UPLOAD_FOLDER, seg_name)

            if os.path.exists(seg_path):
                last_seg_png = seg_path
                used_pngs.add(last_seg_png)
                print(f"Seg PNG saved: {seg_name}")
            else:
                print(f"Seg PNG NOT found: {seg_path}")

            last_ai_frame = frame_idx

        # ---------- Overlay last segmentation ----------
        if last_seg_png and os.path.exists(last_seg_png):
            seg = cv2.imread(last_seg_png)
            if seg is not None:
                seg = cv2.resize(seg, (w, h))
                annotated = cv2.addWeighted(annotated, 0.65, seg, 0.35, 0)

        # ---------- Text overlays ----------
        y = 40
        cv2.putText(annotated, f"Phase: {phase}", (20, y), font, fs, (0,0,255), th)
        y += line
        cv2.putText(annotated, f"Step: {step}", (20, y), font, fs, (255,0,0), th)
        y += line
        act_txt = "Actions: " + (", ".join(actions[:5]) if actions else "none")
        cv2.putText(annotated, act_txt, (20, y), font, fs, (0,255,255), th)

        # Detection probabilities (bottom)
        y = h - 140
        for name, prob in list(last_probs.items())[:6]:
            txt = f"{name}: {prob*100:.1f}%"
            cv2.putText(annotated, txt, (20, y), font, 0.7, (0,255,0), 2)
            y += 28

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  → {frame_idx}/{total}")

    cap.release()
    writer.release()

    import subprocess
    fixed_path = output_path.replace('.mp4', '_fixed.mp4')
    cmd = [
        'ffmpeg', '-i', output_path,
        '-c', 'copy', '-movflags', '+faststart', fixed_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Replace original
    os.replace(fixed_path, output_path)
    print(f"MP4 fixed with faststart: {output_path}")

    # Clean temporary PNGs
    for p in used_pngs:
        try: os.remove(p)
        except: pass

    print(f"Annotated video saved: {output_path}")
    return output_path

# ----------------------------------------------------------------------
#  Serve uploaded / annotated files
# ----------------------------------------------------------------------
@combined_bp.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ----------------------------------------------------------------------
#  Endpoint
# ----------------------------------------------------------------------
@combined_bp.route('/analyze_surgical_video', methods=['POST'])
def analyze_surgical_video_route():
    print("\nCombined analysis STARTED")

    # ----- Auth -----
    auth = request.headers.get('Authorization')
    if not auth:
        return jsonify({'error': 'Missing token'}), 401
    try:
        token = auth.split()[1]
        payload = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user_id = payload['user_id']
    except Exception as e:
        return jsonify({'error': f'Invalid token: {e}'}), 401

    # ----- Upload -----
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if not file or not file.filename:
        return jsonify({'error': 'Invalid file'}), 400

    ts = int(datetime.utcnow().timestamp())
    safe_name = secure_filename(f"{user_id}_{ts}_{file.filename}")
    video_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(video_path)
    output_path = os.path.join(UPLOAD_FOLDER, f"annotated_{safe_name}")

    # ----- Clean old annotated videos (keep only this run) -----
    for old in glob.glob(os.path.join(UPLOAD_FOLDER, "annotated_*.mp4")):
        if os.path.abspath(old) != os.path.abspath(output_path):
            try:
                os.remove(old)
                print(f"Removed old annotated: {old}")
            except: pass

    try:
        # ----- Global inference -----
        phase_step_res = run_phase_step_inference(video_path, PHASE_STEP_MODEL)
        atomic_res     = run_atomic_actions_inference(video_path, ATOMIC_ACTIONS_MODEL)

        # ----- Robust Phase/Step parsing -----
        phase = step = "Unknown"
        if isinstance(phase_step_res, dict):
            # try human-readable keys first
            phase = phase_step_res.get('phase') or phase_step_res.get('predicted_phase') or phase
            step  = phase_step_res.get('step')  or phase_step_res.get('predicted_step')  or step
            # fall back to ID → name mapping
            if 'phase_id' in phase_step_res and phase_id2cat:
                phase = phase_id2cat.get(phase_step_res['phase_id'], phase)
            if 'step_id' in phase_step_res and step_id2cat:
                step = step_id2cat.get(phase_step_res['step_id'], step)
        print(f"Phase/Step raw → {phase_step_res}  →  Phase: {phase} | Step: {step}")

        # ----- Actions parsing -----
        actions = []
        if isinstance(atomic_res, (list, tuple)):
            if atomic_res and isinstance(atomic_res[0], str):
                actions = [a for a in atomic_res if a]
            elif atomic_res and isinstance(atomic_res[0], int):
                actions = [action_names[i] for i in atomic_res if 0 <= i < len(action_names)]
        elif isinstance(atomic_res, dict):
            actions = atomic_res.get('actions', [])
        print(f"Actions raw → {atomic_res}  →  {actions}")

        # ----- Generate annotated video -----
        final_path = generate_annotated_video(
            input_video_path=video_path,
            output_path=output_path,
            phase=phase, step=step, actions=actions,
            detection_model=DETECTION_MODEL,
            segmentation_model=SEGMENTATION_MODEL
        )

        phase_full = phase_name2cat.get(phase, {'name': phase, 'description': "Description unavailable"})
        step_full = step_name2cat.get(step, {'name': step, 'description': "Description unavailable"})

        # ----- DB -----
        db.history.insert_one({
            'user_id': user_id,
            'input_path': video_path,
            'output_path': final_path,
            'result': {'phase': phase_full, 'step': step_full, 'actions': actions},
            'timestamp': datetime.utcnow()
        })

        # ----- Return RELATIVE path (frontend will fetch via /uploads/<file>) -----
        return jsonify({
            'video_path': f"annotated_{safe_name}",
            'analysis': {'phase': phase_full, 'step': step_full, 'actions': actions}
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        for p in (video_path, output_path):
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        return jsonify({'error': f'Analysis failed: {e}'}), 500