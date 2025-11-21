# Robot Perception + Brain (MQTT)

Three small Python services wire a webcam to a vision-language model and a simple rule-based "brain" over MQTT.

## Architecture
- `vision_pub_mqtt.py`: captures frames from the default webcam, downsizes to `TARGET_W x TARGET_H`, JPEG-encodes, base64s, and publishes to `robot/vision`.
- `vision_consumer_mqtt.py`: buffers the last `SEQ_LEN` frames per camera (respecting `FRAME_STRIDE`), runs `LiquidAI/LFM2-VL-450M` via `transformers` to caption the sequence, and publishes results to `robot/brain_context`.
- `brain_consumer_mqtt.py`: consumes captions and runs a small Python rule set to choose an action (swappable with an LLM later).
- Shared settings live in `config.py`.

## Requirements
- Python 3.12
- Mosquitto (or any MQTT broker) reachable at `MQTT_BROKER_HOST`.
- GPU strongly recommended for the LFM2-VL-450M model; CPU is possible but slow. Consider quantization if running on edge hardware.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Mosquitto
- If installed locally: `mosquitto -v`
- Or via Docker: `docker run --rm -p 1883:1883 eclipse-mosquitto`

## Configuration
Edit `config.py` for MQTT host/port, topics, camera ID, sequence length, frame stride, and target frame size.

## Run the services
In separate terminals after activating the venv:
- Vision publisher (webcam -> MQTT): `python vision_pub_mqtt.py`
- Vision consumer (MQTT -> caption -> MQTT): `python vision_consumer_mqtt.py`
- Brain consumer (MQTT -> action): `python brain_consumer_mqtt.py`

Stop any service with `Ctrl+C`.
