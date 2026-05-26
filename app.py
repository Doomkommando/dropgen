"""
DROP.GEN — Backend Flask FINAL
Version complete et corrigee
"""

import os
import json
import time
import threading
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
import anthropic

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INPUT_DIR  = Path("/tmp/dropgen/input")
OUTPUT_DIR = Path("/tmp/dropgen/output")
TEMP_DIR   = Path("/tmp/dropgen/temp")

for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
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


@app.route("/hashtags", methods=["POST"])
def hashtags():
    data   = request.get_json()
    artist = data.get("artist", "artiste")
    style  = data.get("style", "techno")
    desc   = data.get("desc", "")
    lang   = data.get("lang", "EN")
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "Cle API manquante"}), 500
    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    language = "French" if lang == "FR" else "English"
    prompt   = (
        f"You are an expert in Instagram/TikTok marketing for electronic music.\n"
        f"Artist: {artist} | Style: {style} | Bio: {desc or 'Electronic music artist'}\n"
        f"Generate exactly 20 optimised hashtags. Language: {language}.\n"
        f"Reply ONLY with hashtags separated by spaces. No other text."
    )
    response = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    tags = [t for t in response.content[0].text.strip().split() if t.startswith("#")]
    return jsonify({"hashtags": tags})


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Aucun fichier recu"}), 400
    file = request.files["video"]
    import re
    safe_name = file.filename
    safe_name = safe_name.replace("Ø", "O").replace("ø", "o")
    safe_name = re.sub(r"[^\w\s._-]", "", safe_name)
    safe_name = re.sub(r"\s+", "_", safe_name).strip("_") or "video.mp4"
    path = INPUT_DIR / safe_name
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
    music_style = data.get("music_style", "techno")
    artist_desc = data.get("artist_desc", "")
    hook_styles = data.get("hook_styles", ["hype", "mystere", "minimal"])
    threshold   = float(data.get("threshold", 6)) / 10.0
    sec_before  = int(data.get("sec_before", 5))
    sec_after   = int(data.get("sec_after", 20))
    hashtags    = data.get("hashtags", "#techno")
    hook_lang   = data.get("hook_lang", "EN")

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
              hook_styles, threshold, sec_before, sec_after, hashtags, hook_lang),
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
        for mp4 in artist_dir.glob("*_final.mp4"):
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


def run_pipeline(job_id, video_path, artist_name, music_style, artist_desc,
                 hook_styles, threshold, sec_before, sec_after, hashtags, hook_lang="EN"):
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

        # ETAPE 3 — Découpage vidéo
        job["steps"]["3"] = "running"
        log(job_id, "Découpage des clips...", "warn")
        artist_dir = OUTPUT_DIR / sanitize_name(artist_name)
        artist_dir.mkdir(exist_ok=True)
        clip_paths = []
        for i, drop_sec in enumerate(drops):
            start    = max(0, drop_sec - sec_before)
            duration = sec_before + sec_after
            out_path = artist_dir / f"drop_{i+1:02d}_raw.mp4"
            cut_clip(video_path, start, duration, out_path)
            clip_paths.append({
                "index": i+1, "path": out_path,
                "timecode_start": start, "timecode_drop": drop_sec,
                "duration": duration
            })
            log(job_id, f"Clip {i+1}/{len(drops)} cree.", "ok")
        job["stats"]["reels"] = len(clip_paths)
        job["steps"]["3"] = "done"

        # ETAPE 4 — Génération hooks
        job["steps"]["4"] = "running"
        log(job_id, "Generation hooks IA...", "warn")
        reels = []
        for clip in clip_paths:
            log(job_id, f"Hooks drop #{clip['index']}...", "warn")
            hooks = generate_hooks(
                artist_name, music_style, artist_desc,
                hook_styles, clip["timecode_drop"], hashtags, hook_lang
            )
            primary_hook = hooks[0]["text"] if hooks else ""
            final_path   = artist_dir / f"drop_{clip['index']:02d}_final.mp4"
            burn_text(clip["path"], primary_hook, final_path)
            reel_data = {
                "index":          clip["index"],
                "filename":       final_path.name,
                "timecode_start": sec_to_str(clip["timecode_start"]),
                "timecode_drop":  sec_to_str(clip["timecode_drop"]),
                "duration":       clip["duration"],
                "hooks":          hooks,
                "download_url":   f"/download/{artist_dir.name}/{final_path.name}"
            }
            reels.append(reel_data)
            job["reels"].append(reel_data)
            job["stats"]["hooks"] += len(hooks)
        job["steps"]["4"] = "done"

        # ETAPE 5 — Export
        job["steps"]["5"] = "running"
        meta_path = artist_dir / "metadata.json"
        meta_path.write_text(json.dumps({
            "artist": artist_name, "style": music_style,
            "generated_at": datetime.now().isoformat(),
            "total_drops": len(drops), "reels": reels
        }, ensure_ascii=False, indent=2))
        audio_path.unlink(missing_ok=True)
        job["steps"]["5"] = "done"

        job["status"] = "done"
        log(job_id, f"Termine — {len(reels)} Reels prets.", "ok")

    except Exception as e:
        job["status"] = "error"
        log(job_id, f"Erreur : {e}", "err")


# ═════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═════════════════════════════════════════════

def extract_audio(video_path, job_id):
    out    = TEMP_DIR / (video_path.stem + "_audio.wav")
    log(job_id, "FFmpeg extraction audio...", "warn")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "22050", "-f", "wav", str(out)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        log(job_id, f"FFmpeg erreur: {result.stderr[-200:]}", "err")
        raise Exception("FFmpeg audio extraction failed: " + result.stderr[-200:])
    log(job_id, f"Audio extrait : {out.name}", "ok")
    return out


def detect_drops(audio_path, sensitivity=0.6, job_id=None):
    if job_id:
        log(job_id, "Analyse energie audio (FFmpeg)...", "warn")

    # Durée totale
    probe = subprocess.run([
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
    rms_values     = np.array([e[1] for e in energies])
    mean_e, std_e  = np.mean(rms_values), np.std(rms_values)
    threshold_val  = mean_e + std_e * (1.0 - sensitivity * 0.5)
    min_distance   = max(1, int(30 / chunk_size))
    peaks, _       = find_peaks(rms_values, height=threshold_val, distance=min_distance)

    drops  = [int(energies[p][0]) for p in peaks if 60 <= energies[p][0] <= duration - 60]
    result = sorted(drops)
    if job_id:
        log(job_id, f"{len(result)} drops detectes sur {duration:.0f}s", "ok")
    return result


def cut_clip(video_path, start_sec, duration, out_path):
    result = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(max(0, start_sec - 0.5)),
        "-i", str(video_path),
        "-ss", "0.5",
        "-t", str(duration),
        "-vf", "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg cut error: {result.stderr[-400:]}")


def burn_text(video_path, text, out_path):
    safe_text = text.replace("'", "").replace(":", " ").replace('"', "")
    vf = (
        f"drawtext=text='{safe_text}':"
        "fontcolor=white:fontsize=28:borderw=2:bordercolor=black:"
        "x=(w-text_w)/2:y=h*0.82:"
        "box=1:boxcolor=black@0.45:boxborderw=8"
    )
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path), "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg burn error: {result.stderr[-400:]}")


def generate_hooks(artist_name, music_style, artist_desc, hook_styles,
                   drop_timecode, hashtags, hook_lang="EN"):
    if not ANTHROPIC_API_KEY:
        return [{"style": s, "text": f"[Hook {s}]"} for s in hook_styles]
    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    language = "French" if hook_lang == "FR" else "English"
    style_descriptions = {
        "hype":         "Hype and aggressive — emojis, caps",
        "mystere":      "Mysterious — short phrase, suspense",
        "minimal":      "Minimalist — no punctuation, clean",
        "question":     "Direct question to viewer",
        "storytelling": "Sensory micro-storytelling",
        "provoc":       "Playful provocation"
    }
    styles_req = "\n".join([f"- {s}: {style_descriptions.get(s, s)}" for s in hook_styles])
    prompt = (
        f"You are an expert in social media marketing for electronic music.\n"
        f"ARTIST: {artist_name} | STYLE: {music_style} | BIO: {artist_desc or 'Techno artist'}\n"
        f"Generate {len(hook_styles)} hooks for an Instagram/TikTok Reel (25 sec).\n"
        f"STYLES:\n{styles_req}\n"
        f"RULES: max 12 words, language: {language}, add {hashtags}\n"
        f'Reply ONLY in JSON: {{"hooks": [{{"style": "...", "text": "..."}}]}}'
    )
    response = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=500,
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
    print(f"\n✓ DROP.GEN demarre sur le port {PORT}\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
