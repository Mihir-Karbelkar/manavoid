"""
Brain consumer:
- Subscribes to config.BRAIN_CONTEXT_TOPIC.
- Applies simple Python heuristics to captions.
- Logs decided actions; structured to swap in an LLM later.
"""

from __future__ import annotations

import json
import time
from typing import Callable

import paho.mqtt.client as mqtt

import config


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def decide_action(caption: str) -> str:
    """Rule-based action selector; replaceable with an LLM later."""
    text = caption.lower()
    if "person" in text or "human" in text or "people" in text:
        return "Approach the person and greet."
    if "fall" in text or ("lying" in text and "floor" in text) or "fell" in text:
        return "Trigger an alert: possible fall detected."
    if "door" in text:
        return "Move towards the door and check it."
    return "Keep observing."


def create_mqtt_client(handler: Callable[[dict], None]) -> mqtt.Client:
    client = mqtt.Client()

    def on_connect(_client, _userdata, _flags, rc: int):
        if rc == 0:
            log("Connected to MQTT broker.")
            _client.subscribe(config.BRAIN_CONTEXT_TOPIC, qos=0)
            log(f"Subscribed to topic {config.BRAIN_CONTEXT_TOPIC}")
        else:
            log(f"MQTT connection failed with code {rc}")

    def on_message(_client, _userdata, msg: mqtt.MQTTMessage):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            handler(payload)
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to process brain context message: {exc}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(
        config.MQTT_BROKER_HOST,
        config.MQTT_BROKER_PORT,
        config.MQTT_KEEPALIVE,
    )
    return client


def handle_brain_context(payload: dict) -> None:
    caption = payload.get("caption", "")
    action = decide_action(caption)
    camera_id = payload.get("camera_id", "unknown")
    ts = payload.get("ts", "n/a")
    log(f"[{camera_id} @ {ts}] Caption: {caption}")
    log(f"[{camera_id}] Action: {action}")


def main() -> None:
    client = create_mqtt_client(handle_brain_context)
    client.loop_start()
    log("Brain consumer running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("Stopping brain consumer.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
