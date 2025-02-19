import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# TODO: Remove these when S3 is no longer needed.
# Retrieve AWS credentials and configurations from environment variables
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.environ.get(
    "AWS_ENDPOINT_URL", default="https://s3.amazonaws.com"
)
COMPUTE_HORDE_VALIDATION_S3_BUCKET = os.environ.get(
    "COMPUTE_HORDE_VALIDATION_S3_BUCKET"
)

COMPUTE_HORDE_VALIDATOR_HOTKEY = os.environ.get("COMPUTE_HORDE_VALIDATOR_HOTKEY")
COMPUTE_HORDE_EXECUTOR_CLASS = os.environ.get("COMPUTE_HORDE_EXECUTOR_CLASS")
COMPUTE_HORDE_FACILITATOR_URL = os.environ.get("COMPUTE_HORDE_FACILITATOR_URL")
COMPUTE_HORDE_JOB_DOCKER_IMAGE = os.environ.get("COMPUTE_HORDE_JOB_DOCKER_IMAGE")
COMPUTE_HORDE_JOB_NAMESPACE = os.environ.get("COMPUTE_HORDE_JOB_NAMESPACE")
COMPUTE_HORDE_JOB_TIMEOUT = int(os.environ.get("COMPUTE_HORDE_JOB_TIMEOUT", 10 * 60))

COMPUTE_HORDE_VOLUME_TASK_REPO_PATH = "/volume/task_repo.pkl"
COMPUTE_HORDE_VOLUME_MODEL_PATH = "/volume/model"
COMPUTE_HORDE_ARTIFACTS_DIR = "/artifacts"
COMPUTE_HORDE_ARTIFACT_OUTPUT_PATH = os.path.join(
    COMPUTE_HORDE_ARTIFACTS_DIR, "output.pkl"
)
