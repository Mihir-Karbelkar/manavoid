#!/usr/bin/env python3

import time
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "LiquidAI/LFM2-VL-450M"

# How many frames in the sequence
SEQ_LEN = 4
# Use every Nth frame from webcam
FRAME_STRIDE = 10
# Downscale frames for speed
TARGET_W, TARGET_H = 160, 120

QUESTION = (
    "These images are in chronological order from first to last. "
    "Briefly describe what happens across the entire sequence."
)


def make_conversation(images, question: str):
    """
    Build the conversation format LFM2-VL expects, with MULTIPLE images.

    images: list[PIL.Image]
    """
    content = []

    # Add each image in order (1st = earliest, last = latest)
    for img in images:
        content.append({"type": "image", "image": img})

    # Add the text question after the images
    content.append({"type": "text", "text": question})

    return [
        {
            "role": "user",
            "content": content,
        },
    ]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- load model & processor -----
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # ----- warmup (single dummy image) -----
    print("Warming up...")
    dummy = Image.new("RGB", (TARGET_W, TARGET_H), (128, 128, 128))
    warmup_conv = make_conversation([dummy], "Describe this image briefly.")
    warmup_inputs = processor.apply_chat_template(
        warmup_conv,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(device)

    with torch.inference_mode():
        _ = model.generate(**warmup_inputs, max_new_tokens=16)
    print("Warmup done.\n")

    # ----- webcam capture loop -----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # You can keep capture size modest; we'll downscale anyway
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frames = []  # list of PIL images
    skip = 0

    print("Capturing sequence… press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            skip += 1

            # Downscale for speed
            small = cv2.resize(
                frame,
                (TARGET_W, TARGET_H),
                interpolation=cv2.INTER_AREA,
            )

            if skip % FRAME_STRIDE == 0:
                # BGR -> RGB -> PIL
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                frames.append(pil_img)
                print(f"Captured frame {len(frames)}/{SEQ_LEN}")

            if len(frames) == SEQ_LEN:
                print("Got full sequence. Querying LFM2-VL…")

                conversation = make_conversation(frames, QUESTION)

                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(device)

                start = time.time()
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                    )
                end = time.time()

                full_text = processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                )[0]

                # Strip off the "user\n...assistant\n" prefix if present
                if "assistant\n" in full_text:
                    caption = full_text.split("assistant\n", 1)[1].strip()
                else:
                    caption = full_text.strip()

                print("\n--- SEQUENCE DESCRIPTION ---")
                print(caption)
                print(f"\nGeneration time: {end - start:.3f} s")

                # one-shot; remove break if you want continuous sequences
                frames = []
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quitting…")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
