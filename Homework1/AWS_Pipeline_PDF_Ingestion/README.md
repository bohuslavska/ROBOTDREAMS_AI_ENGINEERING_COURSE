# AWS PDF Ingestion Pipeline

## Overview

This project demonstrates an automated AWS pipeline for PDF document processing.

Pipeline architecture:

`PDF → S3 → S3 Event Notification → SQS → Lambda → S3`

When a PDF file is uploaded to the `input/` folder in an S3 bucket, Amazon S3 sends an event notification to an Amazon SQS queue. AWS Lambda reads the message from the queue, downloads the PDF from S3, extracts text using the `pypdf` library, and saves the extracted text back to S3 as a `.txt` file in the `output/` folder.

---

## AWS Services Used

- **Amazon S3** — stores input PDF files and output text files
- **Amazon SQS** — acts as a message queue between S3 and Lambda
- **AWS Lambda** — processes uploaded PDF files automatically
- **IAM** — manages permissions for Lambda to access SQS and S3

---

## Project Flow

1. A PDF file is uploaded to the `input/` folder in the S3 bucket
2. S3 detects a new `.pdf` file and sends an event notification
3. The event is delivered to an SQS queue
4. Lambda is triggered by the SQS queue
5. Lambda reads the queue message and gets the S3 bucket name and object key
6. Lambda downloads the PDF file from S3
7. Lambda extracts text from the PDF using `pypdf`
8. Lambda saves the extracted text as a `.txt` file into the `output/` folder in the same S3 bucket

---

## Example Structure

```text
s3://pdf-ingestion/
├── input/
│   └── sample.pdf
└── output/
    └── sample.txt
```

---

## Lambda Logic

The Lambda function:

- reads SQS messages
- parses the S3 event inside the message
- downloads the PDF from S3
- extracts text page by page
- writes the result back to S3

---

## Python Library

This project uses:

- **pypdf** — for PDF text extraction

---

## Key Implementation Details

- Separate prefixes were used:
  - `input/` for uploaded PDF files
  - `output/` for processed text files
  - This prevents recursive processing of Lambda output files
- Lambda execution role was configured with permissions for:
  - reading messages from SQS
  - reading PDF files from S3
  - writing result files back to S3

---

## Result

The final result is a working event-driven AWS pipeline that automatically processes uploaded PDF documents and stores extracted text in S3 without manual intervention.