"""
Shared configuration for the robot MQTT/vision stack.
Edit these values to match your environment.
"""

MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_KEEPALIVE = 60

VISION_TOPIC = "robot/vision"
BRAIN_CONTEXT_TOPIC = "robot/brain_context"

# Vision pipeline parameters
SEQ_LEN = 4
FRAME_STRIDE = 2  # only keep every Nth frame on the consumer side
TARGET_W = 160
TARGET_H = 120
CAMERA_ID = "cam0"
