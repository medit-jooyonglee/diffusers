import os
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: start_s3.py /path/to/deployment.yaml")
    config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ["S3_PREFIX"].strip("/")
    if not bucket or not prefix:
        raise ValueError("S3_BUCKET and S3_PREFIX must be non-empty")
    for asset in config["assets"]:
        source = f"s3://{bucket}/{prefix}/{asset['prefix'].strip('/')}/"
        print(f"Downloading {source} to {asset['destination']}", flush=True)
        subprocess.run(
            ["aws", "s3", "sync", source, asset["destination"], "--only-show-errors"],
            check=True,
        )
    for required in config["required_files"]:
        if not Path(required).is_file():
            raise FileNotFoundError(f"S3 deployment asset missing: {required}")
    command = [sys.executable, "/app/my/design_tune_server.py", *config["server_args"]]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
