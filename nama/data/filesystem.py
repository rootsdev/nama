import glob as glb
import os
import tempfile
from fnmatch import fnmatch
from urllib.parse import urlparse

import boto3


def parse_s3_path(s3_url):
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def s3_glob(s3_path):
    # Parse the S3 URL
    bucket, prefix = parse_s3_path(s3_path)

    # Split the prefix into directory path and file pattern
    *dir_parts, file_pattern = prefix.split("/")
    dir_prefix = "/".join(dir_parts)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=dir_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if fnmatch(filename, file_pattern):
                yield f"s3://{bucket}/{key}"


def glob(path):
    if path.startswith("s3://"):
        return list(s3_glob(path))
    else:
        return glb.glob(path)


def download_file_from_s3(s3_path, target_path=None):
    bucket, key = parse_s3_path(s3_path)
    # get the s3_path extension
    _, ext = os.path.splitext(key)

    # Create an S3 client
    s3 = boto3.client("s3")

    # Create a temporary file
    if target_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = f"{temp_file.name}{ext}"
        temp_file.close()
        target_path = temp_file_path

    try:
        # Download the file from S3 to the temporary file
        s3.download_file(bucket, key, target_path)
    except Exception as e:
        print(f"Error downloading file {s3_path} from S3: {e}")

    return target_path


def upload_file_to_s3(file_path, s3_path):
    """
    Upload a parquet file to an S3 bucket

    :param file_path: File to upload
    :param s3_path: S3 path
    :return: True if file was uploaded, else False
    """
    bucket, key = parse_s3_path(s3_path)

    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        s3.upload_file(file_path, bucket, key)
    except Exception as e:
        print(f"Error uploading file {s3_path} to S3: {e}")
        return False
    return True


def save_file(out_path, fn):
    _, ext = os.path.splitext(out_path)
    local_out_path = (
        f"{tempfile.NamedTemporaryFile(delete=False).name}{ext}" if out_path.startswith("s3://") else out_path
    )
    fn(local_out_path)
    if out_path.startswith("s3://"):
        upload_file_to_s3(local_out_path, out_path)
