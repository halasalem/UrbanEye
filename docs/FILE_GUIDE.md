# UrbanEye file guide

This guide explains every file in the repository in plain language, then adds the technical detail needed to understand how it fits into the system. Paths below are relative to the repository root.

## Dashboard and main backend

### `UrbanEye_dashboard_voice_v56.html`

**In simple terms:** The control panel a person sees and uses in their browser.

It contains the page structure (HTML), appearance (CSS), and interactive behavior (JavaScript). It sends text commands, voice recordings, and image requests to the local FastAPI backend and displays telemetry, camera imagery, and analysis results. It also integrates Firebase Authentication and Firestore for account and record features.

**Needs:** The local backend and your own Firebase setup. The uploaded Firebase values are placeholders. The separate direct Anthropic plate-scan function is incomplete; a private API key must not be added to browser code.

### `urbaneye_web_backend_voice_v56.py`

**In simple terms:** The coordinator between the control panel and the Python system.

This FastAPI application exposes HTTP endpoints for starting the controller, submitting commands, landing, reading status and logs, receiving voice recordings, serving camera images, and requesting vision analysis. Its controller reuses the navigation code. The active vision pipeline calls `vl_resolver.py` and can forward images to the optional Flask API depending on the backup setting.

**Technical detail:** In this v56 version, browser voice uploads are transcribed directly through Hugging Face requests inside the backend. The website voice route does not call `speech_recognition_hf.py`.

**Needs:** Navigation modules, simulator connections, and `HF_TOKEN` for hosted inference. It normally listens on `127.0.0.1:8000`. Firebase dashboard login does not itself authorize these backend endpoints.

## Navigation and sensors

### `drone_nav_final.py`

**In simple terms:** The flight instruction and navigation manager.

It calculates distances, bearings, and coordinate conversions; sends MAVLink commands and position targets to PX4; and implements takeoff, destination travel, hover movements, rotation, and landing sequences. Background threads read position and ground-distance data and continually send flight targets. It also supports a standalone terminal interface and calls the command/location resolver.

**Needs:** PX4 SITL, MAVLink telemetry, and the expected AirSim vehicle and sensor configuration. The default vehicle is `PX4`, ground sensor is `Distance`, and MAVLink connection is `udp:0.0.0.0:14551`. PX4 remains the flight controller; the language model does not directly drive aircraft motors.

### `object_avoidance_final.py`

**In simple terms:** The helper that watches for obstacles ahead.

It reads the AirSim forward-distance sensor in a background thread, filters sensor readings, and provides movement assistance when obstacles are detected. Navigation uses its obstacle information and helper functions during flight.

**Needs:** AirSim and a compatible forward sensor; the reader's default sensor name is `ForwardDistance`. This is sensor-based avoidance support, not a YOLO image detector or a guarantee of collision-free flight.

### `unreal_camera_reader.py`

**In simple terms:** The helper that keeps checking for the newest camera picture.

It runs a background reader for images exported to disk by Unreal. The backend can use these frames for display and analysis. It reads existing image files; it does not create the Unreal scene or configure Unreal's camera for you.

**Needs:** Unreal exporting camera frames. The default file is `D:\UrbanEyeCamera\urbaneye_latest.png`.

## Language, speech, and hosted vision

### `llm_resolver.py`

**In simple terms:** Turns a sentence into an instruction and helps find the requested place.

It normalizes English and Arabic text, uses a hosted Qwen model for structured command interpretation, and includes rule-based fallback parsing. It can save and match named locations and query OpenStreetMap's Nominatim service for destination coordinates.

**Needs:** `HF_TOKEN` for hosted language-model requests and internet access for hosted inference/geographical lookup. Model/provider and geographical search settings can be configured through environment variables. Its output is interpreted command/location data; navigation executes it.

### `speech_recognition_hf.py`

**In simple terms:** A separate helper for turning a recording into written words.

It uses Hugging Face's `InferenceClient` for hosted speech recognition, with configurable Whisper model/provider settings. It provides standalone/helper functionality for audio files and optional command-queue workflows.

**Needs:** `HF_TOKEN` and an audio input. This is not the active transcription implementation for browser voice uploads in the v56 backend; that route implements its own direct requests.

### `vl_resolver.py`

**In simple terms:** Sends a picture to a hosted AI model and organizes its answer.

It prepares image data, builds a vision prompt, calls the hosted vision-language model, extracts JSON from the reply, and normalizes incidents, bounding boxes, confidence information, and plate text into the format used by the backend. It supports both image bytes and image files.

**Needs:** `HF_TOKEN` and hosted model/provider access. The supplied configuration uses Qwen vision-language inference. This file calls a model; it does not train one. Outputs still require evaluation and can be incorrect.

## Windows launchers

### `Run_UrbanEye_Backend_Website_v56.bat`

**In simple terms:** A Windows shortcut that starts several pieces of UrbanEye.

It switches to its own folder, sets demo environment variables, opens the PX4 launcher, starts Uvicorn for the FastAPI application, and opens the HTML dashboard. It selects hosted vision and explicitly sets the YOLO backup mode to `never`.

**Needs:** An available Python environment, an already configured `HF_TOKEN`, WSL/PX4, and the simulator setup. It does not start Unreal or the optional Flask API.

### `Start_UrbanEye_PX4_v8.bat`

**In simple terms:** Starts the simulated flight controller inside Linux on Windows.

It opens WSL, moves to `~/PX4-Autopilot/`, and runs `make px4_sitl none`, keeping the terminal available for output. The Windows home directory now uses `%USERPROFILE%` instead of one fixed username.

**Needs:** WSL and an existing compatible PX4 installation. It does not install PX4.

## Optional local computer vision API

### `UrbanEye_API/app.py`

**In simple terms:** A separate image-analysis service running on your own computer.

This Flask application loads the incident and plate YOLO weights, detects incidents and plate regions, and uses EasyOCR to read plate crops. Image preprocessing includes resize, contrast/threshold variants, and sharpening; candidate selection supports stacked plates and combines OCR evidence. It exposes health and image-analysis endpoints and returns JSON results.

**Needs:** Both local `.pt` weights in `UrbanEye_API/models/` and its Python dependencies. It uses CUDA when available and CPU otherwise. It loads weights at startup and normally listens on port `5000`. The v56 launcher does not start it automatically.

### `UrbanEye_API/test_api.py`

**In simple terms:** Sends one example picture to the local API to check the response.

It opens `test_image.png` from its own folder, posts it to `http://127.0.0.1:5000/analyze_file`, and prints the HTTP status and returned text.

**Needs:** The running Flask API, model weights, and a local test image. The image is excluded from Git. This is a manual request script, not an automated test suite with assertions.

### `UrbanEye_API/models/README.md`

**In simple terms:** Instructions explaining where the missing trained models belong.

It lists `incident_best.pt` and `plate_best.pt` as local requirements. The weights are intentionally excluded from the repository. This document contains no model data and does not download models.

## Dependencies and local configuration

### `requirements_web.txt`

**In simple terms:** The shopping list of Python packages for the main system.

It lists FastAPI, Uvicorn, Pydantic, pymavlink, AirSim, OpenCV, NumPy, requests, and huggingface_hub. Install it using `python -m pip install -r requirements_web.txt`.

**Remember:** It does not install Unreal, PX4, model weights, or credentials. Package versions are not pinned, so it is not a record of an exactly reproduced environment.

### `UrbanEye_API/requirements.txt`

**In simple terms:** The package list for the local image-analysis service.

It lists Ultralytics, Flask, Flask-CORS, OpenCV, NumPy, EasyOCR, Pillow, and requests. This is separate from the main backend's dependency list because the local vision service is optional.

### `UrbanEye_API/requirements_gpu_nvidia.txt`

**In simple terms:** An alternative package list for an NVIDIA GPU setup.

It includes PyTorch, torchvision, and torchaudio plus the local vision dependencies, using a CUDA wheel index. It is intended for a compatible NVIDIA/CUDA environment; listing these packages does not prove GPU support is available on a particular computer.

### `saved_locations.example.json`

**In simple terms:** An empty template for remembering destinations.

It contains a version number and an empty `locations` list. Copy it to `saved_locations.json` for local use. The resolver can then store names, aliases, GPS coordinates, and timestamps there. The actual personal saved-location file was not uploaded.

### `.env.example`

**In simple terms:** A sample settings sheet with no real token.

It documents environment variable names for Hugging Face access, model/provider choices, and the local vision API address.

**Remember:** The current scripts read shell environment variables. They do not automatically load `.env`; copying this example alone will not configure them. Never replace the example token with a real token in a committed file.

### `.gitignore`

**In simple terms:** The list of local files Git should leave out of future commits.

It excludes tokens, local environment files, model weights, saved locations, audio recordings, debug images, logs, and Python cache/environment folders. It keeps `.env.example` eligible for commit.

**Remember:** It does not erase already committed files or check every file for secrets. Files must still be reviewed before uploading.

## Documentation

### `README.md`

**In simple terms:** The repository's front page.

It introduces UrbanEye, links the demo, outlines the architecture, lists files, explains setup, identifies limitations, and credits the project team. Start here before trying to run the code.

### `docs/FILE_GUIDE.md`

**In simple terms:** The document you are reading: a reference explaining each file.

It expands the short README file list with responsibilities, required inputs/services, and useful technical distinctions. It is documentation and is not executed by Python.

## How the files work together

- **Text:** dashboard sends text to FastAPI; the resolver interprets it; navigation uses the instruction and PX4 executes flight control.
- **Voice:** dashboard sends recorded audio to FastAPI; the backend transcribes it through hosted speech recognition; the resulting text follows the command path.
- **Camera:** Unreal exports an image; the camera reader makes it available to the backend for display or analysis.
- **Hosted vision:** the backend calls `vl_resolver.py`, then returns the normalized results to the dashboard.
- **Optional local vision:** when configured, the backend forwards an image to the Flask API, which runs YOLO/EasyOCR with the separately supplied weights.

This guide describes the supplied code and configuration. It does not assert that every service or simulator configuration has been tested together in the repository-preparation environment.
