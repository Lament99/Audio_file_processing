import base64
import io
import tempfile
import os

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Audio Noise Reducer")


def apply_frequency_filter(data: np.ndarray, sr: int, low_cut: float, high_cut: float) -> np.ndarray:
    """Apply a high-pass, low-pass, or bandpass Butterworth filter."""
    nyq = sr / 2
    low = low_cut / nyq if low_cut > 0 else None
    high = high_cut / nyq if high_cut < nyq else None

    # Clamp to safe range (Butterworth breaks if values hit 0 or 1)
    if low: low = max(0.001, min(low, 0.999))
    if high: high = max(0.001, min(high, 0.999))

    if low and high and low < high:
        b, a = butter(5, [low, high], btype="band")
    elif low:
        b, a = butter(5, low, btype="high")
    elif high:
        b, a = butter(5, high, btype="low")
    else:
        return data  # no filter needed

    return filtfilt(b, a, data)


def build_waveform_image(data: np.ndarray, reduced: np.ndarray, sr: int) -> str:
    """Return a base64-encoded PNG of the before/after waveforms."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
    fig.patch.set_facecolor("#f7f3ec")

    time_axis = np.linspace(0, len(data) / sr, len(data))

    for ax, signal, color, title in [
        (ax1, data,    "#c41e3a", "Original"),
        (ax2, reduced, "#8b1a1a", "Processed"),
    ]:
        ax.plot(time_axis, signal, color=color, linewidth=0.5)
        ax.set_title(title, color="#2c2c2c", fontsize=11, pad=6)
        ax.set_facecolor("#ede8de")
        ax.tick_params(colors="#6b6055")
        for spine in ax.spines.values():
            spine.set_color("#d4cfc5")

    ax2.set_xlabel("Time (s)", color="#6b6055", fontsize=9)
    fig.tight_layout(pad=2.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    mode: str = Form("manual"),
    prop_decrease: float = Form(0.75),
    stationary: bool = Form(False),
    time_constant_s: float = Form(1.0),
    low_cut: float = Form(0.0),
    high_cut: float = Form(20000.0),
    hpss_margin: float = Form(3.0),
):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file.")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(contents)
        tmp_in_path = tmp_in.name

    try:
        data, sr = librosa.load(tmp_in_path, sr=None, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")
    finally:
        os.unlink(tmp_in_path)

    if mode == "auto":
        # HPSS — extract harmonic layer (voice) and discard percussive (noise)
        filtered = librosa.effects.harmonic(data, margin=hpss_margin)
    else:
        # Manual bandpass filter
        filtered = apply_frequency_filter(data, sr, low_cut, high_cut)

    # Step 2 — noise reduction on the filtered signal
    reduced = nr.reduce_noise(
        y=filtered,
        sr=sr,
        prop_decrease=prop_decrease,
        stationary=stationary,
        time_constant_s=time_constant_s,
    )

    waveform_b64 = build_waveform_image(data, reduced, sr)

    wav_buf = io.BytesIO()
    sf.write(wav_buf, reduced, sr, format="WAV")
    wav_buf.seek(0)
    audio_b64 = "data:audio/wav;base64," + base64.b64encode(wav_buf.read()).decode("utf-8")

    return JSONResponse({
        "waveform": waveform_b64,
        "audio": audio_b64,
        "sample_rate": sr,
        "duration": round(len(reduced) / sr, 2),
    })


# Serve frontend — must come after API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")