# Assistive Navigation Prototype

> **Research prototype — not a mobility aid.**
> This project is an early-stage experiment. It is **not** a certified assistive
> device and must **not** be used to guide a person around obstacles. Direction is
> **camera-relative only**: "ahead" means "the middle third of the picture", not
> "close" and not "in your path". The spoken "about N metres" figure is a **rough
> size-based estimate**, not a measurement: it cannot detect depth, closing speed
> or collision risk.

A real-time webcam prototype that uses an already-trained YOLO11 detector to
recognise objects, estimate their left / centre / right position in the frame,
keep a stable identity for each object across frames, and speak short
announcements such as *"chair detected about 2 metres on your left."* The
distance is deliberately approximate — see [Approximate distance](#approximate-distance).

The trained weights are used **read-only**. Nothing here retrains, fine-tunes or
rewrites the model or the dataset.

---

## What it does

```text
Webcam
  ↓
YOLO11 detection (existing best.pt)
  ↓
Bounding boxes + class names + confidence
  ↓
Camera-relative direction (left / centre / right)
  ↓
Approximate distance from box height ("about N metres")
  ↓
ByteTrack identity + lifecycle events
  ↓
Announcement text + cooldown
  ↓
Asynchronous text-to-speech
  ↓
Audio output
```

Each frame is processed independently for detection, while tracking and the
voice queue provide continuity between frames:

- **Detection** — the trained 35-class model runs on every frame.
- **Direction** — the horizontal centre of a box selects one of three zones.
  A small smoothing filter stops boxes on a zone divider from flickering.
- **Distance** — box height plus a typical real-world size gives a *rough*
  estimate, spoken as "about 2 metres". It is not a measurement.
- **Tracking** — ByteTrack (through Ultralytics) keeps a numeric ID per object.
  If ByteTrack is unavailable, a built-in class-aware IoU matcher takes over, so
  the rest of the program always has stable IDs.
- **Voice** — announcements are queued to a worker thread, so a slow speech
  engine never blocks the camera loop. Cooldowns stop the same object from being
  announced on every frame.

---

## Project layout

```text
Assistive-Navigation/
├── models/
│   └── best.pt                 # trained YOLO11 weights (read-only)
├── config/
│   ├── class_names.json        # class ID -> display name (35 entries)
│   ├── object_heights.json     # typical sizes (metres) for distance estimates
│   └── cameras.json            # camera aliases (laptop, droidcam)
├── src/
│   ├── cameras.py              # camera discovery, aliases, backend selection
│   ├── detection.py            # model loading, camera stream, inference
│   ├── direction.py            # left / centre / right estimation + smoothing
│   ├── distance.py             # approximate distance from box height
│   ├── tracking.py             # track identity, lifecycle and direction events
│   ├── voice.py                # sentence building, TTS backends, cooldowns
│   └── main.py                 # pipeline, HUD, keyboard controls, CLI
├── tests/                      # unittest suite
├── checks/
│   ├── make_test_video.py      # builds the synthetic scenario clip
│   └── scenario.mp4            # clip used by the integration test
├── requirements.txt
├── RUN_assistive_nav.bat       # main launcher (pass flags to it)
├── RUN_laptop.bat              # double-click: laptop webcam
├── RUN_droidcam.bat            # double-click: DroidCam
└── README.md
```

---

## Technology stack

| Layer            | Library                                   |
| ---------------- | ----------------------------------------- |
| Detection        | Ultralytics YOLO11 (`best.pt`)            |
| Inference        | PyTorch (CUDA build recommended)          |
| Camera / drawing | OpenCV                                    |
| Tracking         | ByteTrack (via Ultralytics) + `lap`       |
| Speech           | `pyttsx3` (SAPI5 on Windows), SAPI fallback |
| Numerics         | NumPy                                     |
| Concurrency      | `threading` + `queue`                     |

---

## Requirements

- Windows, Linux or macOS (the speech backend is most reliable on Windows).
- Python 3.10+ (tested on 3.11).
- A webcam for live use. A video file works too.
- For real-time GPU inference: an NVIDIA GPU. The reference machine is an
  RTX 4050 Laptop (6 GB) with an Intel Core i7 and 24 GB RAM.

---

## Installation

```bash
# 1. Create / activate an environment (conda or venv)
conda create -n assistive-nav python=3.11 -y
conda activate assistive-nav

# 2. Install PyTorch with CUDA first (pick the build for your driver)
#    Example for CUDA 12.8:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

If you only need CPU inference, install the default `torch` wheel instead.

### Dependencies

| Package          | Why it is needed                                  |
| ---------------- | ------------------------------------------------- |
| `ultralytics`    | YOLO11 inference and ByteTrack                    |
| `torch`          | model execution                                   |
| `opencv-python`  | camera, drawing, video files, screenshots         |
| `numpy`          | array handling                                    |
| `pyttsx3`        | offline speech synthesis                          |
| `lap`            | assignment solver required by ByteTrack           |
| `pywin32`        | *optional* — direct SAPI fallback on Windows      |
| `pygrabber`      | *optional* — DirectShow device names for `--camera laptop` / `droidcam` |

---

## Configure the model weights

The repository ships with:

- `models/best.pt` — a copy of the trained weights;
- `config/class_names.json` — the matching 35 class names.

Both are **copies**. The originals in the training project are untouched.

To use different weights or names, either replace those files or pass paths
explicitly:

```bash
python src/main.py \
  --weights path/to/best.pt \
  --names   path/to/class_names.json
```

`class_names.json` maps numeric IDs to display names. Keys beginning with `_`
are treated as comments. The names are applied to the in-memory model only; the
checkpoint is never modified. Edit the JSON to rename a class — no retraining is
needed.

### Approximate distance

Announcements include a hedged distance such as *"person detected about 2 metres
ahead."* This is computed from the box height, a typical real-world size for the
class and an assumed 60° vertical field of view:

```text
distance ≈ (typical object height × focal length in pixels) / box height in pixels
```

`config/object_heights.json` holds the typical heights in metres, keyed by class
ID (display names work too). `null` disables the estimate for classes whose box
height is meaningless, such as a manhole cover or a flat sign. Edit those values
to match your own objects and camera. The on-screen box label shows the same
figure as `~2m`.

**This is a rough framing cue, not a measurement.** Object sizes vary, phone
cameras distort the edges of the frame, and a partly visible object looks
farther away than it is. Never use it to judge clearance or to navigate. Use
`--no-distance` to restore the simpler "chair detected ahead." wording, and
`--vertical-fov` if your camera's field of view is not about 60 degrees.

Print the names the program will use:

```bash
python src/main.py --list-classes
```

List the installed speech voices:

```bash
python src/main.py --list-voices
```

---

## Run the application

### Windows launchers (no typing)

Double-click any of these in File Explorer:

| File | Camera |
| ---- | ------ |
| `RUN_assistive_nav.bat` | index 0 (laptop webcam) |
| `RUN_laptop.bat` | `--camera laptop` |
| `RUN_droidcam.bat` | `--camera droidcam` |

The window prints a log path; the full run is written to
`checks/logs/last_run.log`. Press `q` inside the video window to stop.

The small launchers are just wrappers, so extra flags still work:

```bat
RUN_droidcam.bat --conf 0.2
```

From a terminal instead, open PowerShell **in this folder** and use the `.\`
prefix (PowerShell does not run programs from the current directory without it):

```powershell
cd "Assistive-Navigation"
.\RUN_assistive_nav.bat --camera droidcam
```

In `cmd.exe` the prefix is not needed:

```bat
cd /d "Assistive-Navigation"
RUN_assistive_nav.bat --camera droidcam
```

> Double-clicking `RUN_assistive_nav.bat` itself **cannot** pass flags — use the
> camera-specific launchers above, or a terminal.

### Camera sources

The camera source is selected with `--camera`. The supported sources are the
laptop webcam and an Android phone through the DroidCam virtual camera. It
accepts an explicit index, an alias, or `auto`:

```bash
# Laptop / built-in webcam (alias)
python src/main.py --camera laptop

# Android phone through the DroidCam Windows virtual camera
python src/main.py --camera droidcam

# First camera that returns a frame
python src/main.py --camera auto

# Explicit OpenCV index (no alias lookup)
python src/main.py --camera 2

# Historical default: index 0
python src/main.py
```

`laptop` and `droidcam` are resolved in this order:

1. an alias in `config/cameras.json`, if present;
2. a DirectShow device-name match (requires the optional `pygrabber` package);
3. a live probe of indices `0..N-1` (`--camera-scan-limit`, default `6`).

The phone-camera index is **never guessed**. If the alias cannot be resolved, the
program stops with the list of detected cameras and the command needed to fix
it. Explicit indices bypass discovery entirely.

To enable name-based alias matching, install the optional helper once:

```bash
pip install pygrabber
```

Without it, aliases come from `config/cameras.json` (as shipped) or from a live
probe; explicit indices always work.

### DroidCam setup

1. Start the DroidCam Client on Windows.
2. Connect the phone over Wi-Fi (or USB) and wait until the client shows the
   live preview.
3. List the cameras and note the DroidCam index:

   ```bash
   python src/main.py --list-cameras
   ```

4. Run with the alias:

   ```bash
   python src/main.py --camera droidcam
   ```

   or double-click `RUN_droidcam.bat`.

If the index changes after a reboot, USB change or driver update, update
`config/cameras.json`:

```json
{
  "aliases": {
    "laptop":    {"index": 0, "backend": "dshow", "name": "Integrated Camera"},
    "droidcam":  {"index": 2, "backend": "auto",  "name": "DroidCam Source 3"}
  }
}
```

`backend` may be `auto` (try the platform order), `dshow`, `msmf`, `any`,
`v4l2` or `avfoundation`. A virtual camera can be available through one backend
but not another; `auto` is the safest default. `--camera-backend` overrides the
config for one run.

The phone IP address is **not** stored or hardcoded. OpenCV sees DroidCam as a
normal Windows camera. Replacing DroidCam later with a native Android stream
only requires another camera source implementation, not a pipeline change.

### A video file

```bash
python src/main.py --source path/to/video.mp4
```

Add `--loop` to repeat the file instead of stopping at the end.

### Useful options

```bash
# Higher confidence threshold, quieter voice
python src/main.py --conf 0.5 --cooldown 10

# Mute at startup, slower speech
python src/main.py --no-voice --rate 150

# Run without a window (e.g. on a headless machine) and log a CSV
python src/main.py --source checks/scenario.mp4 --headless \
  --log-csv run.csv --stats-json stats.json

# Save an annotated video and periodic screenshots
python src/main.py --record --snapshot-every 150
```

### Keyboard controls

| Key       | Action                                      |
| --------- | ------------------------------------------- |
| `q` / Esc | Quit                                        |
| `v`       | Toggle voice feedback on / off              |
| `+` / `=` | Raise the confidence threshold by 0.05      |
| `-`       | Lower the confidence threshold by 0.05      |
| `s`       | Save a screenshot                           |
| `m`       | Stop the current sentence                    |
| `t`       | Toggle tracking (ByteTrack ↔ IoU fallback)   |

---

## What the overlay shows

- Bounding boxes, class name, confidence, smoothed direction and track ID.
- Zone dividers and the labels `LEFT` / `AHEAD` / `RIGHT`.
- FPS, inference time in milliseconds, and the number of visible objects.
- Current confidence threshold, tracking mode and voice status.
- The most recent announcement.
- A camera-reconnect counter if the camera was interrupted.

Box colour follows the zone: orange for left, green for ahead, blue for right.

---

## Example announcements

```text
"chair detected on your left."
"person detected ahead."
"mobile phone detected on your right."
"chair moved from on your left to ahead."
"person, upper body detected ahead."   # parenthesised class name
```

Notes:

- A track is announced when it first appears. While it stays in the same zone it
  is not repeated.
- An object that is already in view when the program starts is announced too:
  the "Assistive navigation ready." greeting does not delay it.
- A direction change produces a "moved from … to …" sentence, subject to its own
  cooldown.
- Per-track cooldowns plus a global gap stop messages from stacking or talking
  over each other.
- Disappearances are tracked and counted but are not spoken by default, to avoid
  a stream of "no longer visible" messages.
- If the speech engine cannot start, the program falls back to a console
  "voice" so the rest of the pipeline still runs.

---

## Configuration reference

### Detection

| Flag               | Default            | Meaning                                   |
| ------------------ | ------------------ | ----------------------------------------- |
| `--weights`        | `models/best.pt`   | trained weights                           |
| `--names`          | `config/class_names.json` | class-name mapping                 |
| `--camera`         | `0`                | index, alias (`laptop`, `droidcam`, `auto`) |
| `--camera-config`  | `config/cameras.json` | alias-to-index mapping                 |
| `--camera-backend` | `auto`             | `auto`, `dshow`, `msmf`, `any`, `v4l2`, `avfoundation` |
| `--camera-scan-limit` | `6`             | indices `0..N-1` probed for an alias      |
| `--camera-width`   | `1280`             | requested capture width                   |
| `--camera-height`  | `720`              | requested capture height                  |
| `--source`         | *(none)*           | video file instead of a camera            |
| `--conf`           | `0.35`             | confidence threshold                      |
| `--iou`            | `0.45`             | NMS IoU threshold                         |
| `--imgsz`          | `640`              | inference size                            |
| `--device`         | auto               | `0` for GPU, `cpu` for CPU                |
| `--no-track`       | off                | disable ByteTrack                         |
| `--max-missing`    | `15`               | frames before a track is dropped          |
| `--stable-frames`  | `3`                | frames before a zone change is accepted   |
| `--left-ratio`     | `1/3`              | left zone edge (fraction of width)        |
| `--right-ratio`    | `2/3`              | right zone edge (fraction of width)       |
| `--object-heights` | `config/object_heights.json` | typical sizes for distance estimates |
| `--vertical-fov`   | `60.0`             | camera vertical field of view (degrees)   |
| `--no-distance`    | off                | disable distance estimates in speech/HUD  |

### Voice

| Flag               | Default   | Meaning                                  |
| ------------------ | --------- | ---------------------------------------- |
| `--no-voice`       | off       | start muted                              |
| `--voice-backend`  | `auto`    | `auto`, `pyttsx3`, `sapi`, `console`     |
| `--voice`          | *(none)*  | substring of the voice name to use       |
| `--rate`           | `175`     | words per minute                         |
| `--volume`         | `1.0`     | `0.0` – `1.0`                            |
| `--cooldown`       | `6.0`     | seconds before the same track repeats    |
| `--min-gap`        | `1.5`     | seconds between any two utterances       |

### Run / diagnostics

| Flag                | Meaning                                       |
| ------------------- | --------------------------------------------- |
| `--headless`        | no window (tests, servers)                    |
| `--max-frames N`    | stop after N frames                           |
| `--loop`            | loop a video file                             |
| `--record`          | write an annotated MP4                        |
| `--snapshot-dir`    | where screenshots / recordings go             |
| `--snapshot-every N`| automatic screenshot every N frames           |
| `--log-csv`         | per-detection CSV log                         |
| `--stats-json`      | summary counters written at shutdown          |
| `--list-classes`    | print the class-name mapping and exit         |
| `--list-cameras`    | probe cameras, print indices and exit         |
| `--list-voices`     | print installed TTS voices and exit           |

---

## Testing

Run the unit and integration tests:

```bash
python -m unittest discover -s tests -v
```

The suite covers:

- left / centre / right zone assignment and boundary behaviour;
- direction smoothing and flicker suppression;
- track appearance, persistence, disappearance, reappearance and class-aware
  matching;
- direction-change events, including the smoothed-direction path;
- announcement wording, priority ordering, cooldowns and the global gap,
  including the greeting not swallowing the first announcement;
- the feed guard that turns a black or frozen camera into a spoken warning;
- distance estimation, JSON height loading and hedged spoken phrases;
- class-name loading and config validation;
- camera spec parsing, alias/config resolution, device-name matching, probe
  de-duplication and release, backend fallback and error messages (all with an
  injected fake capture, so no real camera is needed);
- sequential video delivery, missing-file errors and camera reconnection
  (using an injected fake capture);
- a full 150-frame end-to-end run on `checks/scenario.mp4`.

The suite currently contains **117 tests** (99 pipeline, camera-source and
distance tests + 16 feed-guard tests + 2 greeting-gap tests). The end-to-end test is skipped
automatically if `models/best.pt` or the clip is missing. Rebuild the clip from
real test-set crops with:

```bash
python checks/make_test_video.py
```

The clip deliberately contains one object, three objects across all zones, a
phone that disappears and returns, and a chair that slides left → centre →
right.

---

## Performance notes

Measured on the reference laptop (RTX 4050, 1280×720 capture):

| Metric                    | Value                       |
| ------------------------- | --------------------------- |
| Steady-state frame time   | ~25–40 ms                   |
| Throughput                | ~25–28 FPS                  |
| Warm-up                   | ~7–10 s, once per run       |

The warm-up is not model loading: Ultralytics letterboxes to a rectangular
640-long-side canvas, so the first frame of a given aspect ratio triggers a
one-off cuDNN autotune. The application warms up on the real capture size to move
that cost to startup instead of the first live frame.

The camera is read in a background thread. For live cameras only the newest frame
is kept, so the loop cannot lag behind reality. Video files are delivered
frame-by-frame instead, which makes the integration test deterministic.

DroidCam adds Wi-Fi transport latency and may deliver a lower resolution than the
laptop webcam (640×480 was observed on the reference machine). The HUD still
shows the application's own FPS and inference time, so you can compare sources
directly. A virtual camera may be available through one Windows backend but not
another; the config's `backend: auto` tries DirectShow, then Media Foundation,
then the default backend.

---

## Current limitations

- **Not a mobility aid.** It does not measure distance, depth, closing speed or
  collision risk, and it must not be used to navigate.
- **Approximate distance only.** "About 2 metres" comes from a box height and an
  assumed object size; real distances can be very different.
- **Camera-relative directions only.** "Ahead" is a position in the image, not a
  statement that an object is in the user's path.
- **2-D detections.** Box size also orders simultaneous announcements; converting
  it to metres relies on the height table and camera field of view, not depth.
- **Detection quality depends on the model and lighting.** The weakest classes
  in the training evaluation were spoon, fork and knife; small or partly hidden
  objects are easy to miss.
- **No obstacle-free path planning, no depth estimation and no LLM integration**
  — these are explicitly out of scope for this prototype.
- **Speech is intentionally terse and rate-limited.** It favours not talking over
  itself over narrating every change.
- **Single camera, single process.** There is no multi-camera fusion and no
  persistence of tracks across restarts.
- **Camera indices and backends are not permanent.** Windows can renumber
  devices after a reboot, USB change or driver update. Re-run
  `--list-cameras` and update `config/cameras.json` when that happens.
- **Name-based aliases need the optional `pygrabber` package.** Without it,
  aliases resolve from `config/cameras.json` or a live probe; explicit indices
  always work.
- **DroidCam is a virtual webcam, not a native Android stream.** Its quality,
  latency and resolution depend on the Wi-Fi link, the phone and the DroidCam
  client. The pipeline itself is source-agnostic, so a native stream can replace
  it later.
- **Audio output is best-effort.** If no TTS engine is available, announcements
  go to the console.

---

## Troubleshooting

| Symptom                              | Fix                                                                 |
| ------------------------------------ | ------------------------------------------------------------------- |
| `Could not open camera ...`          | Close Teams / Zoom / Camera; check Windows camera privacy settings; run `--list-cameras` and pass the correct index. |
| `Camera alias 'droidcam' was not detected` | Start DroidCam and wait for the preview; run `--list-cameras`; set the index in `config/cameras.json` or pass `--camera <index>`. |
| DroidCam opens but the feed is black | The app now says "Camera is not sending video" and logs a warning. Open the DroidCam phone app and wait for the preview in DroidCam Client, check the Wi-Fi link, uncover the lens, then try another DroidCam source or `--camera-backend msmf`. |
| DroidCam shows a still image | The app says "Camera video is frozen". Open DroidCam Client on Windows, wait for the phone preview, and check the Wi-Fi link; or use `--camera laptop` to confirm the pipeline. |
| Startup line plays, then no announcements | The camera feed is black or frozen (see above) or nothing in view belongs to the 35 classes. Run `--camera laptop` and point at a person to check the pipeline, or lower `--conf` to `0.2`. |
| Distance sounds wrong | It is an estimate from box height, not a sensor. Tune `config/object_heights.json` for your objects and check `--vertical-fov`; use `--no-distance` to hide it. |
| Alias works after a reboot but not later | Windows renumbered the camera. Re-run `--list-cameras` and update `config/cameras.json`. |
| `Weights not found`                  | Put `best.pt` in `models/` or pass `--weights`.                     |
| No speech, console messages only     | Run `--list-voices`; try `--voice-backend sapi`; check the volume.  |
| ByteTrack falls back to IoU          | Install `lap` (`pip install lap`). The program keeps working either way. |
| First frame is very slow             | Expected: cuDNN autotune for the capture aspect ratio. Later frames are fast. |
| `half` deprecation warning           | Older Ultralytics builds; the code detects `quantize` vs `half` automatically. |

---

## License / provenance

The detector was trained separately. `models/best.pt` and
`config/class_names.json` are copies; the training project and dataset were not
modified by this prototype.
