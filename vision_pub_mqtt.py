"""
Webcam publisher:
- Captures frames from the default webcam.
- Downscales and JPEG-encodes them.
- Publishes base64-encoded frames to MQTT topic config.VISION_TOPIC.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from typing import Optional

import cv2
import paho.mqtt.client as mqtt

import config


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def create_mqtt_client() -> tuple[mqtt.Client, threading.Event]:
    connected_event = threading.Event()
    client = mqtt.Client()

    def on_connect(_client: mqtt.Client, _userdata, _flags, rc: int) -> None:
        if rc == 0:
            log("Connected to MQTT broker.")
            connected_event.set()
        else:
            log(f"MQTT connection failed with code {rc}.")

    client.on_connect = on_connect
    client.connect(
        config.MQTT_BROKER_HOST,
        config.MQTT_BROKER_PORT,
        config.MQTT_KEEPALIVE,
    )
    client.loop_start()
    return client, connected_event


def encode_frame(frame) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main() -> None:
    log("Starting vision publisher.")
    client, connected = create_mqtt_client()
    if not connected.wait(timeout=10):
        log("MQTT connection timeout.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("Could not open webcam (device 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.TARGET_W * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TARGET_H * 2)
    log("Webcam opened; publishing frames. Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log("Failed to read frame from webcam.")
                time.sleep(0.5)
                continue

            small = cv2.resize(
                frame,
                (config.TARGET_W, config.TARGET_H),
                interpolation=cv2.INTER_AREA,
            )

            image_b64 = encode_frame(small)
            if image_b64 is None:
                log("JPEG encoding failed; skipping frame.")
                continue

            payload = {
                "camera_id": config.CAMERA_ID,
                "ts": time.time(),
                "image_b64": image_b64,
            }
            client.publish(config.VISION_TOPIC, json.dumps(payload), qos=0)
            time.sleep(0.05)
    except KeyboardInterrupt:
        log("Stopping vision publisher.")
    finally:
        cap.release()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
