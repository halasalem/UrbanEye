# UrbanEye

**An autonomous drone simulation for traffic monitoring with multilingual voice control.**

UrbanEye connects a web dashboard to a simulated PX4 drone, combining natural language command interpretation, geographical destination resolution, telemetry, camera imagery, and AI-assisted incident and license plate analysis.

This repository contains the Python services and dashboard from the undergraduate graduation project. The Unreal Engine environment, PX4 installation, trained YOLO weights, credentials, and private saved locations are not included.

## Demo

[Watch the UrbanEye demo on YouTube](https://www.youtube.com/watch?v=X64fSH44iSk)

## Features

- English and Arabic command interpretation using a hosted language model.
- Voice input using hosted Whisper speech recognition.
- Drone takeoff, navigation, hover, rotation, and landing controls.
- Saved destination lookup and OpenStreetMap/Nominatim geographical resolution.
- Flight telemetry and camera display in the dashboard.
- Vision-language analysis of incidents and license plates.
- An optional local YOLO and EasyOCR service for incident detection and plate reading.
- Firebase Authentication and Firestore integration in the dashboard.

## Architecture

| Component | Role |
| --- | --- |
| Unreal Engine, Cesium, AirSim/Colosseum | Simulation environment, scene imagery, and sensor interface |
| PX4 SITL and pymavlink | Flight control and telemetry |
| FastAPI backend | Dashboard commands, status, voice uploads, and vision orchestration |
| Hugging Face hosted models | Command interpretation, speech recognition, and vision-language analysis |
| Optional Flask API | Local YOLO detection and EasyOCR |
| HTML dashboard and Firebase | User interface, authentication, and stored application records |

The dashboard connects to FastAPI at `127.0.0.1:8000`. The optional local computer vision API listens on port `5000`. Flight navigation uses MAVLink at `udp:0.0.0.0:14551` and expects the configured AirSim vehicle and sensors.

## Repository contents

| File | Purpose |
| --- | --- |
| `drone_nav_final.py` | Drone navigation and flight control |
| `object_avoidance_final.py` | Obstacle avoidance support |
| `llm_resolver.py` | Command parsing and destination resolution |
| `speech_recognition_hf.py` | Hosted speech recognition helper |
| `vl_resolver.py` | Hosted vision-language analysis |
| `unreal_camera_reader.py` | Reads camera frames exported by Unreal |
| `urbaneye_web_backend_voice_v56.py` | FastAPI backend |
| `UrbanEye_dashboard_voice_v56.html` | Web dashboard |
| `Run_UrbanEye_Backend_Website_v56.bat` | Windows launcher for PX4, backend, and dashboard |
| `Start_UrbanEye_PX4_v8.bat` | WSL PX4 SITL launcher |
| `UrbanEye_API/` | Optional local computer vision API and dependencies |
| `saved_locations.example.json` | Empty template for local saved destinations |
| `.env.example` | Environment variable reference without credentials |

## Setup

### Prerequisites

The project was developed with Python 3.10 on Windows and PX4 SITL in WSL. Prepare a compatible Unreal Engine/Cesium/AirSim or Colosseum environment and PX4 setup separately. Their project files, sensor configuration, and installation are not supplied here.

The navigation configuration expects an AirSim vehicle named `PX4` and a distance sensor named `Distance`. The camera reader defaults to `D:\UrbanEyeCamera\urbaneye_latest.png`; configure Unreal to export frames there, or adjust the camera path in the source for your setup.

### Python environment

Run from the repository root in Windows Command Prompt:

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements_web.txt
copy saved_locations.example.json saved_locations.json
```

The copied saved-locations file is local and ignored by Git. Do not overwrite an existing personal saved-locations file when updating your working project.

### Hosted models

Set your own token before starting the backend:

```bat
set HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
```

The scripts read environment variables directly. `.env.example` documents the options; copying it to `.env` alone does not load those values.

The supplied v56 launcher selects hosted Whisper and `Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud` for vision. Command parsing defaults to `Qwen/Qwen2.5-7B-Instruct` with the `featherless-ai` provider. Model/provider access must be configured on your own account. These identifiers are the uploaded code's configuration and have not been verified against current provider availability.

### Firebase dashboard

Replace the `YOUR_...` values in `firebaseConfig` inside the HTML with your own Firebase web app configuration. Configure Authentication, Firestore, authorized domains, and appropriate database access rules separately. Existing Firebase records and rules are not included. The Firebase web configuration identifies a client application; server credentials and service-account keys must remain private.

### Launch

1. Prepare PX4 SITL and the Unreal simulation. The WSL launcher expects PX4 in `~/PX4-Autopilot/` and runs `make px4_sitl none`.
2. From the activated Python environment, run `Run_UrbanEye_Backend_Website_v56.bat`. It starts PX4 and FastAPI and opens the HTML dashboard.
3. Start Unreal Play and confirm that MAVLink, AirSim sensors, and exported camera frames are available before issuing commands.

Alternatively, start FastAPI manually after preparing the simulation:

```bat
python -m uvicorn urbaneye_web_backend_voice_v56:app --host 127.0.0.1 --port 8000
```

Health endpoint: `http://127.0.0.1:8000/api/health`.

### Optional local YOLO/EasyOCR API

The v56 launcher uses hosted vision and sets `URBANEYE_YOLO_BACKUP_MODE=never`; it does not start the local API.

To use the local API, place your separately held weights at:

- `UrbanEye_API/models/incident_best.pt`
- `UrbanEye_API/models/plate_best.pt`

Then install and run:

```bat
python -m pip install -r UrbanEye_API\requirements.txt
python UrbanEye_API\app.py
```

`requirements_gpu_nvidia.txt` provides an alternative NVIDIA CUDA dependency configuration. The API uses CUDA when available and otherwise uses CPU. The weights are required at API startup.

For backend fallback, set `URBANEYE_YOLO_BACKUP_MODE=smart` or `always` before a manual backend launch. The supplied launcher explicitly sets `never`.

To run the standalone request script, supply a suitable local image at `UrbanEye_API/test_image.png`, start the API, and run `python UrbanEye_API/test_api.py`. That script sends an image to `/analyze_file`; it is not an automated test suite.

## Limitations and validation

This is a research simulation prototype. Model outputs can be incorrect and depend on image quality, trained weights, and hosted service availability. No accuracy or benchmark results are asserted here.

The dashboard includes an additional plate-image scan function that calls Anthropic directly without authentication. That function is incomplete in the supplied code. Do not embed a private API key in browser code; completing it requires a backend integration.

The local APIs are intended for the development simulation environment. Dashboard login does not establish backend authorization. A deployment would need additional authentication and access controls.

For this repository preparation, Python syntax and embedded credential patterns were checked. Full drone operation, Firebase access, hosted inference, and GPU/CPU inference were not run in the preparation environment.

## Project credits

Developed by **Hala Salem, Lama AbuTuaimeh, and Suliman Abusamak** at **Applied Science University**, supervised by **Dr. Bashar Alshouha**.

This repository preserves the team project's source while providing an independent repository under Hala Salem's account.
