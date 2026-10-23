import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = ROOT / "docker/design-tune-server/start_s3.py"


class DesignTuneS3Test(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.required = root / "model" / "model_index.json"
        self.required.parent.mkdir()
        self.required.write_text("{}")
        self.config = root / "deployment.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "assets": [{"prefix": "models/klein-4b", "destination": str(self.required.parent)}],
                    "required_files": [str(self.required)],
                    "server_args": ["--model", str(self.required.parent), "--fp8-weights"],
                }
            )
        )
        self.environment = patch.dict(os.environ, {"S3_BUCKET": "test-bucket", "S3_PREFIX": "/releases/v1/"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def run_start(self):
        with patch.object(sys, "argv", [str(START_SCRIPT), str(self.config)]):
            runpy.run_path(str(START_SCRIPT), run_name="__main__")

    def test_downloads_before_replacing_process(self):
        with patch("subprocess.run") as download, patch("os.execv") as execute:
            self.run_start()
        download.assert_called_once_with(
            [
                "aws",
                "s3",
                "sync",
                "s3://test-bucket/releases/v1/models/klein-4b/",
                str(self.required.parent),
                "--only-show-errors",
            ],
            check=True,
        )
        execute.assert_called_once_with(
            sys.executable,
            [sys.executable, "/app/my/design_tune_server.py", "--model", str(self.required.parent), "--fp8-weights"],
        )

    def test_failed_download_prevents_server_start(self):
        error = subprocess.CalledProcessError(1, ["aws", "s3", "sync"])
        with patch("subprocess.run", side_effect=error), patch("os.execv") as execute:
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_start()
        execute.assert_not_called()

    def test_missing_asset_prevents_server_start(self):
        self.required.unlink()
        with patch("subprocess.run"), patch("os.execv") as execute:
            with self.assertRaisesRegex(FileNotFoundError, "S3 deployment asset missing"):
                self.run_start()
        execute.assert_not_called()

    def test_empty_release_prefix_prevents_download(self):
        os.environ["S3_PREFIX"] = "/"
        with patch("subprocess.run") as download, patch("os.execv") as execute:
            with self.assertRaisesRegex(ValueError, "must be non-empty"):
                self.run_start()
        download.assert_not_called()
        execute.assert_not_called()
