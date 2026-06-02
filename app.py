"""
DROP.GEN v5 — Detection FFmpeg par chunks, zero RAM
"""

import os, re, json, time, threading, subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS

INPUT_DIR  = Path("/tmp/dropgen/input")
OUTPUT_DIR = Path("/tmp/dropgen/output")
TEMP_DIR   = Path("/tmp/dropgen/temp")
for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PORT = int(os.environ.get("PORT", 5000))
app  = Flask(__name__, static_folder=".")
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024
jobs = {}

@app.route("/")
def index():
    return send_file("reels_pipeline_ui.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Aucun fichier recu"}), 400
    file = request.files["video"]
    safe = re.sub(r"[^\w._-]", "_", file.filename.replace("Ø","O").replace("ø","o")).strip("_") or "video.mp4"
    path = INPUT_DIR / safe
    file.save(str(path))
    return jsonify({"status":"uploaded","filename":safe,"size_mb":round(path.stat().st_size/1024/1024,1)})

@app.route("/generate", methods=["POST"])
def generate():
    data       = request.get_json()
    filename   = data.get("filename")
    artist     = data.get("artist_name","Artiste")
    threshold  = float(data.get("threshold",6)) / 10.0
    sec_before = int(data.get("sec_before",5))
    sec_after  = int(data.get("sec_after",20))
    if not filename:
        return jsonify({"error":"Fichier manquant"}), 400
    video_path = INPUT_DIR / filename
    if not video_path.exists():
        return jsonify({"error":"Fichier introuvable"}), 404
    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {"status":"running","steps":{},"drops":[],"reels":[],"logs":[],"stats":{"drops":0,"reels":0}}
    threading.Thread(target=run_pipeline, args=(job_id,video_path,artist,threshold,sec_before,sec_after), daemon=True).start()
    return jsonify({"job_id":job_id})

@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error":"Job inconnu"}), 404
    return jsonify(jobs[job_id])

@app.route("/download/<path:filepath>")
def download(filepath):
    return send_from_directory(str(OUTPUT_DIR), filepath, as_attachment=True)

@app.route("/download-zip/<artist_name>")
def download_zip(artist_name):
    import zipfile, io
    artist_dir = OUTPUT_DIR / sanitize(artist_name)
    if not artist_dir.exists():
        return jsonify({"error":"Dossier introuvable"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(artist_dir.glob("drop_*.mp4")):
            zf.write(f, f.name)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="application/zip",
        headers={"Content-Disposition":f"attachment; filename={artist_name}_reels.zip"})


@app.route("/recut", methods=["POST"])
def recut():
    data       = request.get_json()
    filename   = data.get("filename")
    artist     = data.get("artist_name","Artiste")
    idx        = int(data.get("index", 1))
    start_sec  = int(data.get("start_sec", 0))
    duration   = int(data.get("duration", 25))

    video_path = INPUT_DIR / filename
    if not video_path.exists():
        return jsonify({"error":"Fichier introuvable"}), 404

    artist_dir = OUTPUT_DIR / sanitize(artist)
    artist_dir.mkdir(exist_ok=True)
    out = artist_dir / f"drop_{idx:02d}.mp4"

    r = subprocess.run([
        "ffmpeg","-y",
        "-ss",str(start_sec),
        "-i",str(video_path),
        "-t",str(duration),
        "-c","copy",
        "-avoid_negative_ts","make_zero",
        str(out)
    ], capture_output=True, text=True)

    if r.returncode != 0:
        return jsonify({"error": r.stderr[-200:]}), 500

    return jsonify({
        "status": "ok",
        "download_url": f"/download/{artist_dir.name}/{out.name}"
    })

def log(job_id, msg, level="info"):
    t = datetime.now().strftime("%H:%M:%S")
    jobs[job_id]["logs"].append({"time":t,"msg":msg,"level":level})
    print(f"[{t}] {msg}")

def run_pipeline(job_id, video_path, artist, threshold, sec_before, sec_after):
    job = jobs[job_id]
    try:
        # STEP 1 — Duration
        job["steps"]["1"] = "running"
        log(job_id,"Analyse de la video...","warn")
        probe = subprocess.run([
            "ffprobe","-v","quiet","-print_format","json","-show_format",str(video_path)
        ], capture_output=True, text=True)
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        log(job_id,f"Duree : {duration:.0f}s","ok")
        job["steps"]["1"] = "done"

        # STEP 2 — Drops
        job["steps"]["2"] = "running"
        log(job_id,"Detection des drops...","warn")
        drops = detect_drops(video_path, duration, threshold, job_id)
        job["drops"] = drops
        job["stats"]["drops"] = len(drops)
        job["steps"]["2"] = "done"
        log(job_id,f"{len(drops)} drops detectes.","ok")

        # STEP 3 — Decoupe
        job["steps"]["3"] = "running"
        artist_dir = OUTPUT_DIR / sanitize(artist)
        artist_dir.mkdir(exist_ok=True)
        reels = []
        for i, drop_sec in enumerate(drops):
            start = max(0, drop_sec - sec_before)
            dur   = sec_before + sec_after
            out   = artist_dir / f"drop_{i+1:02d}.mp4"
            log(job_id,f"Clip {i+1}/{len(drops)} — {sec_to_str(start)}...","warn")
            r = subprocess.run([
                "ffmpeg","-y",
                "-ss",str(start),
                "-i",str(video_path),
                "-t",str(dur),
                "-c","copy",
                "-avoid_negative_ts","make_zero",
                str(out)
            ], capture_output=True, text=True)
            if r.returncode != 0:
                raise Exception(f"FFmpeg: {r.stderr[-200:]}")
            reel = {
                "index":i+1,"filename":out.name,
                "timecode_start":sec_to_str(start),
                "timecode_drop":sec_to_str(drop_sec),
                "duration":dur,
                "download_url":f"/download/{artist_dir.name}/{out.name}"
            }
            reels.append(reel)
            job["reels"].append(reel)
            job["stats"]["reels"] = len(reels)
            log(job_id,f"Clip {i+1} OK.","ok")
        job["steps"]["3"] = "done"

        job["steps"]["4"] = "running"
        job["steps"]["4"] = "done"
        job["status"] = "done"
        log(job_id,f"Termine — {len(reels)} clips prets.","ok")

    except Exception as e:
        job["status"] = "error"
        log(job_id,f"Erreur : {e}","err")


def detect_drops(video_path, duration, sensitivity=0.6, job_id=None):
    """
    Detection des drops via analyse directe de la video par chunks de 3s.
    Mesure l'energie des basses frequences (kick techno) sans charger en RAM.
    Detecte le pattern breakdown -> drop (chute d'energie suivie d'une montee).
    """
    from scipy.signal import find_peaks

    chunk = 3
    energies = []
    t = 0

    if job_id: log(job_id,"Analyse energie par chunks...","warn")

    while t < duration:
        # Analyse frequentielle directement sur la video (pas besoin d'extraire l'audio)
        r = subprocess.run([
            "ffmpeg","-ss",str(t),"-t",str(chunk),
            "-i",str(video_path),
            "-af","lowpass=f=200,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-vn","-f","null","-"
        ], capture_output=True, text=True)

        rms = -60.0
        for line in r.stderr.splitlines():
            if "RMS_level" in line and "=" in line:
                try:
                    v = float(line.split("=")[-1].strip())
                    if v > -100: rms = v
                except: pass
        energies.append((t, rms))
        t += chunk

    if not energies: return []

    rms_vals = np.array([e[1] for e in energies])

    # Lissage sur 3 points
    smoothed = np.convolve(rms_vals, np.ones(3)/3, mode='same')

    # Normalisation
    rms_min = np.min(smoothed)
    rms_max = np.max(smoothed)
    if rms_max - rms_min < 1.0:
        if job_id: log(job_id,"Signal trop uniforme, ajustement...","warn")
    normalized = (smoothed - rms_min) / (rms_max - rms_min + 1e-6)

    # Detection pattern DROP TECHNO :
    # 1. Cherche les creux (breakdowns) — energie basse
    # 2. Verifie qu'il y a une montee brutale dans les 20s apres

    drops = []
    min_gap = int(25 / chunk)  # 25s minimum entre deux drops
    last_drop_idx = -min_gap

    for i in range(2, len(normalized)-4):
        # Critere 1 : creux local (breakdown)
        is_low = normalized[i] < 0.35
        if not is_low:
            continue

        # Critere 2 : montee brutale dans les 5-20s suivantes
        search_window = min(i + int(20/chunk), len(normalized)-1)
        segment = normalized[i:search_window]
        if len(segment) < 2:
            continue

        max_after = np.max(segment)
        delta = max_after - normalized[i]

        # Le drop = montee d'au moins 0.25 de l'amplitude normalisee
        min_delta = 0.35 - (sensitivity * 0.05)
        if delta < min_delta:
            continue

        # Position du pic apres le creux
        peak_offset = np.argmax(segment)
        drop_idx = i + peak_offset
        drop_t = int(energies[drop_idx][0])

        if drop_t < 60 or drop_t > duration - 60:
            continue
        if drop_idx - last_drop_idx < min_gap:
            continue

        drops.append(drop_t)
        last_drop_idx = drop_idx

    if job_id: log(job_id,f"{len(drops)} drops detectes (pattern breakdown->drop).","ok")
    return sorted(drops)


def sanitize(name):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

def sec_to_str(sec):
    h,m,s = sec//3600,(sec%3600)//60,sec%60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

if __name__ == "__main__":
    print(f"\n✓ DROP.GEN v5 port {PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
