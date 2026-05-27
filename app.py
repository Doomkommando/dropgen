"""
DROP.GEN v2 — Backend minimaliste
Import video → Detection drops → Découpage 9:16 → Téléchargement
"""

import os
import json
import time
import re
import threading
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INPUT_DIR  = Path("/tmp/dropgen/input")
OUTPUT_DIR = Path("/tmp/dropgen/output")
TEMP_DIR   = Path("/tmp/dropgen/temp")

for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__, static_folder=".")
CORS(app)

jobs = {}


# ═════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════

@app.route("/")
def index():
    return send_file("reels_pipeline_ui.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Aucun fichier recu"}), 400
    file      = request.files["video"]
    safe_name = file.filename
    safe_name = safe_name.replace("Ø", "O").replace("ø", "o")
    safe_name = re.sub(r"[^\w\s._-]", "", safe_name)
    safe_name = re.sub(r"\s+", "_", safe_name).strip("_") or "video.mp4"
    path      = INPUT_DIR / safe_name
    file.save(str(path))
    return jsonify({
        "status":   "uploaded",
        "filename": safe_name,
        "size_mb":  round(path.stat().st_size / 1024 / 1024, 1)
    })


@app.route("/generate", methods=["POST"])
def generate():
    data        = request.get_json()
    filename    = data.get("filename")
    artist_name = data.get("artist_name", "Artiste")
    threshold   = float(data.get("threshold", 6)) / 10.0
    sec_before  = int(data.get("sec_before", 5))
    sec_after   = int(data.get("sec_after", 20))

    if not filename:
        return jsonify({"error": "Fichier manquant"}), 400
    video_path = INPUT_DIR / filename
    if not video_path.exists():
        return jsonify({"error": f"Fichier {filename} introuvable"}), 404

    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {
        "status": "running", "steps": {}, "drops": [],
        "reels": [], "logs": [],
        "stats": {"drops": 0, "reels": 0}
    }

    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, video_path, artist_name, threshold, sec_before, sec_after),
        daemon=True
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify(jobs[job_id])


@app.route("/debug/<job_id>")
def debug(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job inconnu"}), 404
    job = jobs[job_id]
    return jsonify({"status": job["status"], "steps": job["steps"], "logs": job["logs"]})


@app.route("/download/<path:filepath>")
def download(filepath):
    return send_from_directory(str(OUTPUT_DIR), filepath, as_attachment=True)


@app.route("/download-zip/<artist_name>")
def download_zip(artist_name):
    import zipfile, io
    artist_dir = OUTPUT_DIR / sanitize_name(artist_name)
    if not artist_dir.exists():
        return jsonify({"error": "Dossier artiste introuvable"}), 404
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for mp4 in sorted(artist_dir.glob("drop_*.mp4")):
            zf.write(mp4, mp4.name)
    zip_buffer.seek(0)
    return Response(
        zip_buffer.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename={artist_name}_reels.zip"}
    )


# ═════════════════════════════════════════════
# PIPELINE
# ═════════════════════════════════════════════

def log(job_id, msg, level="info"):
    t = datetime.now().strftime("%H:%M:%S")
    jobs[job_id]["logs"].append({"time": t, "msg": msg, "level": level})
    print(f"[{t}] [{level.upper()}] {msg}")


def run_pipeline(job_id, video_path, artist_name, threshold, sec_before, sec_after):
    job = jobs[job_id]
    try:
        # ETAPE 1 — Extraction audio
        job["steps"]["1"] = "running"
        log(job_id, "Extraction audio...", "warn")
        audio_path = extract_audio(video_path, job_id)
        job["steps"]["1"] = "done"

        # ETAPE 2 — Detection des drops
        job["steps"]["2"] = "running"
        log(job_id, "Detection des drops...", "warn")
        drops = detect_drops(audio_path, sensitivity=threshold, job_id=job_id)
        job["drops"] = drops
        job["stats"]["drops"] = len(drops)
        job["steps"]["2"] = "done"
        log(job_id, f"{len(drops)} drops detectes.", "ok")

        # ETAPE 3 — Découpage vidéo 9:16
        job["steps"]["3"] = "running"
        log(job_id, "Decoupage clips 9:16...", "warn")
        artist_dir = OUTPUT_DIR / sanitize_name(artist_name)
        artist_dir.mkdir(exist_ok=True)

        reels = []
        for i, drop_sec in enumerate(drops):
            start    = max(0, drop_sec - sec_before)
            duration = sec_before + sec_after
            out_path = artist_dir / f"drop_{i+1:02d}.mp4"
            log(job_id, f"Clip {i+1}/{len(drops)} — {sec_to_str(start)}...", "warn")
            cut_clip(video_path, start, duration, out_path)
            reel_data = {
                "index":          i + 1,
                "filename":       out_path.name,
                "timecode_start": sec_to_str(start),
                "timecode_drop":  sec_to_str(drop_sec),
                "duration":       duration,
                "download_url":   f"/download/{artist_dir.name}/{out_path.name}"
            }
            reels.append(reel_data)
            job["reels"].append(reel_data)
            job["stats"]["reels"] = len(reels)
            log(job_id, f"Clip {i+1} OK.", "ok")

        job["steps"]["3"] = "done"

        # ETAPE 4 — Export metadata
        job["steps"]["4"] = "running"
        meta_path = artist_dir / "metadata.json"
        meta_path.write_text(json.dumps({
            "artist":       artist_name,
            "generated_at": datetime.now().isoformat(),
            "total_drops":  len(drops),
            "reels":        reels
        }, ensure_ascii=False, indent=2))
        audio_path.unlink(missing_ok=True)
        job["steps"]["4"] = "done"

        job["status"] = "done"
        log(job_id, f"Termine — {len(reels)} clips prets a telecharger.", "ok")

    except Exception as e:
        job["status"] = "error"
        log(job_id, f"Erreur : {e}", "err")


# ═════════════════════════════════════════════
# FONCTIONS
# ═════════════════════════════════════════════

def extract_audio(video_path, job_id):
    out    = TEMP_DIR / (video_path.stem + "_audio.wav")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "22050", "-f", "wav", str(out)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("FFmpeg audio failed: " + result.stderr[-300:])
    log(job_id, f"Audio extrait : {out.name}", "ok")
    return out


def detect_drops(audio_path, sensitivity=0.6, job_id=None):
    if job_id:
        log(job_id, "Analyse energie audio...", "warn")

    # Durée totale
    probe    = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(audio_path)
    ], capture_output=True, text=True)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if job_id:
        log(job_id, f"Duree : {duration:.0f}s", "ok")

    # Analyse RMS par chunks de 5 secondes
    chunk_size = 5
    energies   = []
    t          = 0
    while t < duration:
        result = subprocess.run([
            "ffmpeg", "-ss", str(t), "-t", str(chunk_size),
            "-i", str(audio_path),
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null", "-"
        ], capture_output=True, text=True)
        rms = -60.0
        for line in result.stderr.splitlines():
            if "RMS_level" in line and "=" in line:
                try:
                    val = float(line.split("=")[-1].strip())
                    if val > -100:
                        rms = val
                except Exception:
                    pass
        energies.append((t, rms))
        t += chunk_size

    if not energies:
        return []

    from scipy.signal import find_peaks
    rms_values    = np.array([e[1] for e in energies])
    mean_e        = np.mean(rms_values)
    std_e         = np.std(rms_values)
    threshold_val = mean_e + std_e * (1.0 - sensitivity * 0.5)
    min_distance  = max(1, int(30 / chunk_size))
    peaks, _      = find_peaks(rms_values, height=threshold_val, distance=min_distance)

    drops  = [int(energies[p][0]) for p in peaks if 60 <= energies[p][0] <= duration - 60]
    result = sorted(drops)
    if job_id:
        log(job_id, f"{len(result)} drops detectes.", "ok")
    return result


def cut_clip(video_path, start_sec, duration, out_path):
    result = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(max(0, start_sec - 0.5)),
        "-i", str(video_path),
        "-ss", "0.5",
        "-t", str(duration),
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg cut error: {result.stderr[-400:]}")


def sanitize_name(name):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


def sec_to_str(sec):
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


if __name__ == "__main__":
    print(f"\n✓ DROP.GEN v2 demarre sur le port {PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
