#!/usr/bin/env python3
"""Delete every object under R2_PREFIX in the bucket (or the whole bucket if no prefix)."""

import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET     = os.environ["R2_BUCKET"]
R2_PREFIX     = "dsec"   # set to "" to wipe the entire bucket

DELETE_WORKERS = 16
DELETE_BATCH   = 1000    # S3 delete_objects max

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.eu.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    region_name="auto",
)


def delete_batch(keys: list[str]) -> int:
    s3.delete_objects(
        Bucket=R2_BUCKET,
        Delete={"Objects": [{"Key": k} for k in keys], "Quiet": True},
    )
    return len(keys)


def main():
    print(f"Listing objects in s3://{R2_BUCKET}/{R2_PREFIX} ...")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=R2_BUCKET, Prefix=R2_PREFIX)

    batches = []
    current = []
    total_listed = 0

    for page in pages:
        for obj in page.get("Contents", []):
            current.append(obj["Key"])
            total_listed += 1
            if len(current) == DELETE_BATCH:
                batches.append(current)
                current = []

    if current:
        batches.append(current)

    if not total_listed:
        print("Nothing to delete.")
        return

    print(f"Found {total_listed} objects across {len(batches)} batches. Deleting...")

    deleted = 0
    with ThreadPoolExecutor(max_workers=DELETE_WORKERS) as pool:
        futures = [pool.submit(delete_batch, b) for b in batches]
        for fut in as_completed(futures):
            deleted += fut.result()
            print(f"  deleted {deleted}/{total_listed}", end="\r")

    print(f"\nDone. {deleted} objects deleted.")


if __name__ == "__main__":
    main()
