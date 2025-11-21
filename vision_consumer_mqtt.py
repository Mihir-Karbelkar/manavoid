"""
MQTT consumer for vision frames:
- Subscribes to config.VISION_TOPIC.
- Buffers the last SEQ_LEN frames per camera (respecting FRAME_STRIDE).
- Runs a vision-language model to caption the sequence.
- Publishes captions to config.BRAIN_CONTEXT_TOPIC.
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Iterable, List, Tuple

import paho.mqtt.client as mqtt
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import config

MODEL_ID = "LiquidAI/LFM2-VL-450M"
QUESTION = (
    "These images are in chronological order from first to last. "
    "Briefly describe what happens across the entire sequence."
)


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def make_conversation(images: Iterable[Image.Image], question: str):
    """Build the chat template payload expected by LFM2-VL with multiple images."""
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def load_model() -> tuple[AutoProcessor, AutoModelForImageTextToText, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading model {MODEL_ID} on device {device}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    dummy = Image.new("RGB", (config.TARGET_W, config.TARGET_H), (128, 128, 128))
    warmup_conv = make_conversation([dummy], "Describe this image briefly.")
    warmup_inputs = processor.apply_chat_template(
        warmup_conv,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(device)
    log("Running warmup inference...")
    with torch.inference_mode():
        _ = model.generate(**warmup_inputs, max_new_tokens=8)
    log("Warmup complete.")
    return processor, model, device


def decode_message(payload: bytes) -> Tuple[str, float, Image.Image]:
    data = json.loads(payload.decode("utf-8"))
    camera_id = data["camera_id"]
    ts = float(data["ts"])
    image_bytes = base64.b64decode(data["image_b64"])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return camera_id, ts, img


def create_mqtt_client(frame_queue: queue.Queue) -> mqtt.Client:
    client = mqtt.Client()

    def on_connect(_client, _userdata, _flags, rc: int):
        if rc == 0:
            log("Connected to MQTT broker.")
            _client.subscribe(config.VISION_TOPIC, qos=0)
            log(f"Subscribed to topic {config.VISION_TOPIC}")
        else:
            log(f"MQTT connection failed with code {rc}")

    def on_message(_client, _userdata, msg: mqtt.MQTTMessage):
        try:
            camera_id, ts, img = decode_message(msg.payload)
            frame_queue.put((camera_id, ts, img))
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to handle incoming message: {exc}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(
        config.MQTT_BROKER_HOST,
        config.MQTT_BROKER_PORT,
        config.MQTT_KEEPALIVE,
    )
    return client


def caption_sequence(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    device: str,
    images: List[Image.Image],
    question: str,
) -> str:
    conversation = make_conversation(images, question)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    full_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "assistant\n" in full_text:
        return full_text.split("assistant\n", 1)[1].strip()
    return full_text.strip()


def frame_worker(
    frame_queue: queue.Queue,
    client: mqtt.Client,
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    device: str,
):
    buffers: dict[str, deque[Tuple[float, Image.Image]]] = defaultdict(
        lambda: deque(maxlen=config.SEQ_LEN)
    )
    counters: dict[str, int] = defaultdict(int)

    while True:
        item = frame_queue.get()
        if item is None:
            log("Frame worker shutting down.")
            break

        camera_id, ts, img = item
        counters[camera_id] += 1
        if counters[camera_id] % config.FRAME_STRIDE != 0:
            continue

        buf = buffers[camera_id]
        buf.append((ts, img))
        if len(buf) < config.SEQ_LEN:
            continue

        images = [im for _, im in buf]
        latest_ts = buf[-1][0]
        try:
            start = time.time()
            caption = caption_sequence(processor, model, device, images, QUESTION)
            duration = time.time() - start
            log(
                f"Captioned sequence for {camera_id} (len={len(images)}) "
                f"in {duration:.2f}s: {caption}"
            )
            payload = {
                "camera_id": camera_id,
                "ts": latest_ts,
                "caption": caption,
                "vision_model": MODEL_ID,
            }
            client.publish(config.BRAIN_CONTEXT_TOPIC, json.dumps(payload), qos=0)
        except Exception as exc:  # noqa: BLE001
            log(f"Sequence caption failed for {camera_id}: {exc}")


def main() -> None:
    processor, model, device = load_model()
    frame_queue: queue.Queue = queue.Queue(maxsize=64)
    client = create_mqtt_client(frame_queue)
    worker = threading.Thread(
        target=frame_worker,
        args=(frame_queue, client, processor, model, device),
        daemon=True,
    )
    worker.start()

    log("Vision consumer is running. Ctrl+C to stop.")
    client.loop_start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("Stopping vision consumer.")
    finally:
        frame_queue.put(None)
        worker.join(timeout=5)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
