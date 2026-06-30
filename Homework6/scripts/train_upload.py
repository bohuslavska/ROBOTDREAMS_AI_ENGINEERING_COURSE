"""
Upload train.jsonl and start an OpenAI fine-tuning job for gpt-4o-mini.

Prints job id; poll separately via:
    python3 scripts/poll_ft.py <job_id>
"""

import os
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent.parent
TRAIN_FILE = HERE / "data" / "train.jsonl"


def main():
    client = OpenAI()

    print(f"Uploading {TRAIN_FILE} ({TRAIN_FILE.stat().st_size} bytes)…")
    file_obj = client.files.create(file=open(TRAIN_FILE, "rb"), purpose="fine-tune")
    print(f"  file id: {file_obj.id}")

    print("Starting fine-tuning job: gpt-4o-mini-2024-07-18, suffix='email-extractor-v1'…")
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model="gpt-4o-mini-2024-07-18",
        suffix="email-extractor-v1",
    )
    print(f"  job id: {job.id}")
    print(f"  status: {job.status}")
    print()
    print("Poll with:")
    print(f"  python3 scripts/poll_ft.py {job.id}")


if __name__ == "__main__":
    main()
