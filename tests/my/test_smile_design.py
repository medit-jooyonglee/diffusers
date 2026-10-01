import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from my.smile_design.embedding_library import EmbeddingLibrary
from my.smile_design.inference import resolve_layout_and_images
from my.smile_design.prompt_templates import build_prompt_entries, load_experiment_config


def save_embedding(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_embeds": torch.full((1, 2, 1), float(value))}, path)


class SmileDesignTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        entries = {}

        def add(entry_id, value):
            relative = f"embeddings/{entry_id.replace('/', '_')}.pt"
            save_embedding(root, relative, value)
            entries[entry_id] = {"file": relative, "prompt": entry_id}

        add("patient_only/base", 5)
        add("patient_only/preset/natural", 5)
        add("patient_only/anchor/whitening/000", 5)
        add("patient_only/anchor/whitening/100", 7)
        add("patient_only/anchor/smile_intensity/000", 5)
        add("patient_only/anchor/smile_intensity/100", 8)
        add("patient_only/grid/whitening_smile/000_000", 0)
        add("patient_only/grid/whitening_smile/100_000", 10)
        add("patient_only/grid/whitening_smile/000_100", 100)
        add("patient_only/grid/whitening_smile/100_100", 110)
        save_embedding(root, "negative.pt", -1)
        manifest = {
            "schema_version": 1,
            "template_version": "test",
            "embedding_shape": [1, 2, 1],
            "negative_embedding": "negative.pt",
            "controls": {
                "whitening": {"default": 0, "anchors": {"0": "low", "100": "high"}},
                "smile_intensity": {"default": 0, "anchors": {"0": "low", "100": "high"}},
            },
            "grids": {"whitening_smile": {"attributes": ["whitening", "smile_intensity"]}},
            "layouts": {"patient_only": {"roles": ["patient"]}},
            "entries": entries,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.library = EmbeddingLibrary(root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_anchor_interpolation(self):
        actual = self.library.interpolate_anchor("patient_only", "whitening", 25)
        self.assertTrue(torch.equal(actual, torch.full((1, 2, 1), 5.5)))

    def test_bilinear_grid_interpolation(self):
        actual = self.library.interpolate_grid(
            "patient_only", "whitening_smile", {"whitening": 50, "smile_intensity": 50}
        )
        self.assertTrue(torch.equal(actual, torch.full((1, 2, 1), 55.0)))

    def test_delta_composition(self):
        actual = self.library.compose_deltas(
            "patient_only", {"whitening": 100, "smile_intensity": 100}
        )
        self.assertTrue(torch.equal(actual, torch.full((1, 2, 1), 10.0)))

    def test_prompt_config_generates_expected_grid(self):
        config, fingerprint = load_experiment_config("my/configs/smile_design/smile_v1.yaml")
        entries = build_prompt_entries(config, layouts=["patient_only"], kinds={"grid"})
        self.assertEqual(len(entries), 25)
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all(entry.prompt.startswith("Edit the dental photograph") for entry in entries))

    def test_out_of_range_slider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between"):
            self.library.interpolate_anchor("patient_only", "whitening", 101)

    def test_patient_only_layout_inference(self):
        patient = Path(self.temporary_directory.name) / "patient.png"
        Image.new("RGB", (16, 16)).save(patient)
        args = type(
            "Arguments",
            (),
            {
                "layout": None,
                "patient": patient,
                "tooth_reference": None,
                "mouth_reference": None,
                "pose_reference": None,
                "extra_reference": None,
            },
        )()
        layout, images, records = resolve_layout_and_images(args, self.library)
        self.assertEqual(layout, "patient_only")
        self.assertEqual(len(images), 1)
        self.assertEqual(records[0]["role"], "patient")


if __name__ == "__main__":
    unittest.main()
