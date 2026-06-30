"""Poll OpenAI fine-tuning job status until it finishes."""

import sys
import time
from openai import OpenAI


def main():
    job_id = sys.argv[1]
    client = OpenAI()

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"[{time.strftime('%H:%M:%S')}] status={job.status}  trained_tokens={job.trained_tokens}  fine_tuned_model={job.fine_tuned_model}")
        if job.status in ("succeeded", "failed", "cancelled"):
            print()
            print(f"FINAL status: {job.status}")
            print(f"fine_tuned_model: {job.fine_tuned_model}")
            print(f"trained_tokens: {job.trained_tokens}")
            print(f"hyperparameters: {job.hyperparameters}")
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
