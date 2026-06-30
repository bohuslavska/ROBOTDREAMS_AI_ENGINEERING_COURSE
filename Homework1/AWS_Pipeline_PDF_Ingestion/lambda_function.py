import json
import os
import urllib.parse
from io import BytesIO

import boto3
from pypdf import PdfReader

s3 = boto3.client("s3")

OUTPUT_PREFIX = "output/"

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n\n".join(pages_text).strip()

def lambda_handler(event, context):
    results = []

    for record in event["Records"]:
        body = json.loads(record["body"])

        # S3 event notification can contain multiple Records
        for s3_record in body["Records"]:
            bucket = s3_record["s3"]["bucket"]["name"]
            key = urllib.parse.unquote_plus(s3_record["s3"]["object"]["key"])

            if not key.startswith("input/") or not key.lower().endswith(".pdf"):
                continue

            response = s3.get_object(Bucket=bucket, Key=key)
            pdf_bytes = response["Body"].read()

            extracted_text = extract_text_from_pdf(pdf_bytes)

            base_name = key.split("/")[-1].rsplit(".", 1)[0]
            output_key = f"{OUTPUT_PREFIX}{base_name}.txt"

            s3.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=extracted_text.encode("utf-8"),
                ContentType="text/plain; charset=utf-8"
            )

            results.append({
                "input_key": key,
                "output_key": output_key,
                "status": "processed"
            })

    return {
        "statusCode": 200,
        "body": json.dumps(results)
    }
