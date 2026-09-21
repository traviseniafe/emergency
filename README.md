# Emergency Detection

A web application that listens and watches for signs of an emergency (dangerous sounds, fire and smoke, distress words) and raises an alert.

**Status:** in progress. Phase 0 is done: the backend and frontend are connected. Detection models are coming next.

## How it fits together

```
Browser (frontend)  <--HTTP/JSON-->  FastAPI backend  -->  detection models  -->  alerts
   what you see                        the API              audio, vision,        email
                                                            speech
```

## Run it

Requires Python 3.10 or newer. From the project root:

```
python3 -m pip install -r requirements.txt
python3 -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 in your browser. Interactive API docs are at http://127.0.0.1:8000/docs.

## Run the tests

```
python3 -m pytest
```

## Roadmap

- [ ] Phase 0: backend, frontend and tests connected
- [ ] Phase 1: train the audio emergency-sound model
- [ ] Phase 2: audio analysis through the API, with microphone and file upload in the browser
- [ ] Phase 3: fire and smoke detection from camera frames
- [ ] Phase 4: speech keyword detection
- [ ] Phase 5: decision logic, event history and email alerts
- [ ] Phase 6: Docker and documentation
