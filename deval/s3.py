import uuid

import bittensor as bt
import boto3
from dotenv import load_dotenv, find_dotenv

from deval.compute_horde_settings import (
    AWS_ACCESS_KEY_ID,
    AWS_ENDPOINT_URL,
    AWS_SECRET_ACCESS_KEY,
    COMPUTE_HORDE_VALIDATION_S3_BUCKET,
)

load_dotenv(find_dotenv())


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=AWS_ENDPOINT_URL,
    )


def get_public_url(key: str) -> str:
    return f"{AWS_ENDPOINT_URL}/{COMPUTE_HORDE_VALIDATION_S3_BUCKET}/{key}"


def upload_data_to_s3(data: bytes) -> str | None:
    data_sample_name = str(uuid.uuid4())
    s3_client = get_s3_client()

    s3_client.put_object(
        Body=data,
        Bucket=COMPUTE_HORDE_VALIDATION_S3_BUCKET,
        Key=data_sample_name,
    )

    # return the public url
    public_url = get_public_url(data_sample_name)
    bt.logging.info(f"Uploaded data sample to {public_url}")
    return public_url
