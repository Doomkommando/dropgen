"""
DROP.GEN v3 — Backend minimaliste
Import video -> Detection drops -> Decoupe -> Telechargement
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
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10 Go
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

def log(job_id, msg, level="info"):
    t = datetime.now().strftime("%H:%M:%S")
    jobs[job_id]["logs"].append({"time":t,"msg":msg,"level":level})
    print(f"[{t}] {msg}")

def run_pipeline(job_id, video_path, artist, threshold, sec_before, sec_after):
    job = jobs[job_id]
    try:
        # STEP 1 — Audio
        job["steps"]["1"] = "running"
        log(job_id,"Extraction audio...","warn")
        audio = extract_audio(video_path, job_id)
        job["steps"]["1"] = "done"

        # STEP 2 — Drops
        job["steps"]["2"] = "running"
        log(job_id,"Detection des drops...","warn")
        drops = detect_drops(audio, threshold, job_id)
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
                "ffmpeg","-y","-i",str(video_path),
                "-ss",str(start),"-t",str(dur),
                "-c","copy",str(out)
            ], capture_output=True, text=True)
            if r.returncode != 0:
                raise Exception(f"FFmpeg: {r.stderr[-200:]}")
            reel = {"index":i+1,"filename":out.name,
                    "timecode_start":sec_to_str(start),
                    "timecode_drop":sec_to_str(drop_sec),
                    "duration":dur,
                    "download_url":f"/download/{artist_dir.name}/{out.name}"}
            reels.append(reel)
            job["reels"].append(reel)
            job["stats"]["reels"] = len(reels)
            log(job_id,f"Clip {i+1} OK.","ok")
        job["steps"]["3"] = "done"

        # STEP 4 — Export
        job["steps"]["4"] = "running"
        audio.unlink(missing_ok=True)
        job["steps"]["4"] = "done"
        job["status"] = "done"
        log(job_id,f"Termine — {len(reels)} clips prets.","ok")

    except Exception as e:
        job["status"] = "error"
        log(job_id,f"Erreur : {e}","err")

def extract_audio(video_path, job_id):
    out = TEMP_DIR / (video_path.stem + "_audio.wav")
    r = subprocess.run([
        "ffmpeg","-y","-i",str(video_path),
        "-vn","-ac","1","-ar","22050","-f","wav",str(out)
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise Exception("Audio failed: " + r.stderr[-200:])
    log(job_id,f"Audio extrait.","ok")
    return out

def detect_drops(audio_path, sensitivity=0.6, job_id=None):
    if job_id: log(job_id,"Analyse energie audio...","warn")
    probe = subprocess.run([
        "ffprobe","-v","quiet","-print_format","json","-show_format",str(audio_path)
    ], capture_output=True, text=True)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if job_id: log(job_id,f"Duree : {duration:.0f}s","ok")

    chunk = 5
    energies = []
    t = 0
    while t < duration:
        r = subprocess.run([
            "ffmpeg","-ss",str(t),"-t",str(chunk),"-i",str(audio_path),
            "-af","astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f","null","-"
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
    from scipy.signal import find_peaks
    rms_vals = np.array([e[1] for e in energies])
    mean, std = np.mean(rms_vals), np.std(rms_vals)
    thresh = mean + std * (1.0 - sensitivity * 0.5)
    peaks, _ = find_peaks(rms_vals, height=thresh, distance=max(1,int(30/chunk)))
    drops = sorted([int(energies[p][0]) for p in peaks if 60 <= energies[p][0] <= duration-60])
    if job_id: log(job_id,f"{len(drops)} drops detectes.","ok")
    return drops

def sanitize(name):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

def sec_to_str(sec):
    h,m,s = sec//3600,(sec%3600)//60,sec%60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

if __name__ == "__main__":
    print(f"\n✓ DROP.GEN v3 port {PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
