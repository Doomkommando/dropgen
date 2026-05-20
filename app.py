"""
DROP.GEN — Backend Flask (version Railway/SaaS)
Remplace app.py par ce fichier pour le déploiement en ligne.
"""

import os
import json
import time
import threading
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

import librosa
import anthropic

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
INPUT_DIR  = Path("/tmp/dropgen/input")
OUTPUT_DIR = Path("/tmp/dropgen/output")
TEMP_DIR   = Path("/tmp/dropgen/temp")

for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Clé API lue depuis les variables d'environnement Railway (jamais dans le code !)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__, static_folder=".")
CORS(app)

# Jobs en mémoire
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
        return jsonify({"error": "Aucun fichier reçu"}), 400
    file = request.files["video"]
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    path = INPUT_DIR / safe_name
    file.save(str(path))
    return jsonify({
        "status": "uploaded",
        "filename": safe_name,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 1)
    })


@app.route("/generate", methods=["POST"])
def generate():
    data        = request.get_json()
    filename    = data.get("filename")
    artist_name = data.get("artist_name", "Artiste")
    music_style = data.get("music_style", "techno")
    artist_desc = data.get("artist_desc", "")
    hook_styles = data.get("hook_styles", ["hype", "mystere", "minimal"])
    threshold   = float(data.get("threshold", 6)) / 10.0
    sec_before  = int(data.get("sec_before", 5))
    sec_after   = int(data.get("sec_after", 20))
    hashtags    = data.get("hashtags", "#techno")

    if not filename:
        return jsonify({"error": "Fichier manquant"}), 400

    video_path = INPUT_DIR / filename
    if not video_path.exists():
        return jsonify({"error": f"Fichier {filename} introuvable"}), 404

    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {
        "status": "running", "steps": {}, "drops": [],
        "reels": [], "logs": [],
        "stats": {"drops": 0, "reels": 0, "hooks": 0}
    }

    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, video_path, artist_name, music_style, artist_desc,
              hook_styles, threshold, sec_before, sec_after, hashtags),
        daemon=True
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify(jobs[job_id])


@app.route("/download/<path:filepath>")
def download(filepath):
    return send_from_directory(str(OUTPUT_DIR), filepath, as_attachment=True)


# ═════════════════════════════════════════════
# PIPELINE
# ═════════════════════════════════════════════

def log(job_id, msg, level="info"):
    t = datetime.now().strftime("%H:%M:%S")
    jobs[job_id]["logs"].append({"time": t, "msg": msg, "level": level})
    print(f"[{t}] [{level.upper()}] {msg}")


def run_pipeline(job_id, video_path, artist_name, music_style, artist_desc,
                 hook_styles, threshold, sec_before, sec_after, hashtags):
    job = jobs[job_id]
    try:
        job["steps"]["1"] = "running"
        log(job_id, "Extraction audio...")
        audio_path = extract_audio(video_path)
        job["steps"]["1"] = "done"

        job["steps"]["2"] = "running"
        log(job_id, "Détection des drops...")
        drops = detect_drops(audio_path, sensitivity=threshold)
        job["drops"] = drops
        job["stats"]["drops"] = len(drops)
        job["steps"]["2"] = "done"
        log(job_id, f"{len(drops)} drops détectés.", "ok")

        job["steps"]["3"] = "running"
        artist_dir = OUTPUT_DIR / sanitize_name(artist_name)
        artist_dir.mkdir(exist_ok=True)
        clip_paths = []
        for i, drop_sec in enumerate(drops):
            start = max(0, drop_sec - sec_before)
            duration = sec_before + sec_after
            out_path = artist_dir / f"drop_{i+1:02d}_raw.mp4"
            cut_clip(video_path, start, duration, out_path)
            clip_paths.append({"index": i+1, "path": out_path,
                                "timecode_start": start, "timecode_drop": drop_sec,
                                "duration": duration})
        job["stats"]["reels"] = len(clip_paths)
        job["steps"]["3"] = "done"

        job["steps"]["4"] = "running"
        reels = []
        for clip in clip_paths:
            log(job_id, f"Hooks drop #{clip['index']}...")
            hooks = generate_hooks(artist_name, music_style, artist_desc,
                                   hook_styles, clip["timecode_drop"], hashtags)
            primary_hook = hooks[0]["text"] if hooks else ""
            final_path = artist_dir / f"drop_{clip['index']:02d}_final.mp4"
            burn_text(clip["path"], primary_hook, final_path)
            reel_data = {
                "index": clip["index"],
                "filename": final_path.name,
                "timecode_start": sec_to_str(clip["timecode_start"]),
                "timecode_drop": sec_to_str(clip["timecode_drop"]),
                "duration": clip["duration"],
                "hooks": hooks,
                "download_url": f"/download/{artist_dir.name}/{final_path.name}"
            }
            reels.append(reel_data)
            job["reels"].append(reel_data)
            job["stats"]["hooks"] += len(hooks)
        job["steps"]["4"] = "done"

        job["steps"]["5"] = "running"
        meta_path = artist_dir / "metadata.json"
        meta_path.write_text(json.dumps({
            "artist": artist_name, "style": music_style,
            "generated_at": datetime.now().isoformat(),
            "total_drops": len(drops), "reels": reels
        }, ensure_ascii=False, indent=2))
        job["steps"]["5"] = "done"
        audio_path.unlink(missing_ok=True)

        job["status"] = "done"
        log(job_id, f"Terminé — {len(reels)} Reels prêts.", "ok")

    except Exception as e:
        job["status"] = "error"
        log(job_id, f"Erreur : {e}", "err")


def extract_audio(video_path):
    out = TEMP_DIR / (video_path.stem + "_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "22050", "-f", "wav", str(out)
    ], check=True, capture_output=True)
    return out


def detect_drops(audio_path, sensitivity=0.6):
    from scipy.signal import find_peaks
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    mean_e, std_e = np.mean(onset_env), np.std(onset_env)
    height = mean_e + std_e * (3.0 - sensitivity * 2.0)
    distance_frames = int(sr / hop * 30)
    peaks, _ = find_peaks(onset_env, height=height, distance=distance_frames)
    duration = librosa.get_duration(y=y, sr=sr)
    drops = [int(librosa.frames_to_time(p, sr=sr, hop_length=hop)) for p in peaks]
    return sorted([t for t in drops if 60 <= t <= duration - 60])


def cut_clip(video_path, start_sec, duration, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(duration),
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k", str(out_path)
    ], check=True, capture_output=True)


def burn_text(video_path, text, out_path):
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    vf = (f"drawtext=text='{safe_text}':fontcolor=white:fontsize=48:"
          "borderw=3:bordercolor=black:x=(w-text_w)/2:y=h*0.12:enable='between(t,0,5)'")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy", str(out_path)
    ], check=True, capture_output=True)


def generate_hooks(artist_name, music_style, artist_desc, hook_styles, drop_timecode, hashtags):
    if not ANTHROPIC_API_KEY:
        return [{"style": s, "text": f"[Hook {s} — clé API manquante]"} for s in hook_styles]
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    style_descriptions = {
        "hype": "Hype et agressif — émojis, majuscules",
        "mystere": "Mystérieux — phrase courte, suspense",
        "minimal": "Minimaliste — sans ponctuation, épuré",
        "question": "Question directe au spectateur",
        "storytelling": "Micro-storytelling sensoriel",
        "provoc": "Provocation décalée"
    }
    styles_req = "\n".join([f"- {s}: {style_descriptions.get(s, s)}" for s in hook_styles])
    prompt = f"""Tu es expert en marketing pour artistes de musique électronique.
ARTISTE : {artist_name} | STYLE : {music_style} | BIO : {artist_desc or "Artiste techno"}
Génère {len(hook_styles)} hooks pour un Reel Instagram/TikTok (25 sec).
STYLES : {styles_req}
RÈGLES : max 12 mots, français ou anglais, ajoute {hashtags}
Réponds UNIQUEMENT en JSON : {{"hooks": [{{"style": "...", "text": "..."}}]}}"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw).get("hooks", [])


def sanitize_name(name):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


def sec_to_str(sec):
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


if __name__ == "__main__":
    print(f"\n✓ DROP.GEN démarré sur le port {PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
