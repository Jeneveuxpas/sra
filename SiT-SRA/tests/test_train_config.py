import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path


diffusers = types.ModuleType("diffusers")
diffusers_models = types.ModuleType("diffusers.models")
diffusers_models.AutoencoderKL = object
diffusers.models = diffusers_models
sys.modules.setdefault("diffusers", diffusers)
sys.modules.setdefault("diffusers.models", diffusers_models)

from train import parse_args


class TrainConfigTest(unittest.TestCase):
    config_path = Path(__file__).parents[1] / "configs" / "sit-b2-sra-cls.yaml"

    def test_loads_sit_b_paper_recipe_and_cls_schedule(self):
        args = parse_args(["--config", str(self.config_path)])

        self.assertEqual(args.model, "SiT-B/2")
        self.assertEqual((args.block_out_s, args.block_out_t), (3, 8))
        self.assertEqual(args.loss_type, "sml1")
        self.assertEqual(args.align_weight, 0.2)
        self.assertEqual(args.max_train_steps, 400_000)
        self.assertEqual(args.batch_size * 8, 256)
        self.assertEqual(args.ema_decay, 0.9999)
        self.assertEqual(args.data_dir, "/dev/shm/data")
        self.assertEqual(args.cls_prob_decay_steps, 100_000)

    def test_explicit_cli_arguments_override_yaml(self):
        args = parse_args(
            [
                "--config",
                str(self.config_path),
                "--batch-size",
                "7",
                "--no-fused-attn",
                "--exp-name",
                "override",
            ]
        )

        self.assertEqual(args.batch_size, 7)
        self.assertFalse(args.fused_attn)
        self.assertEqual(args.exp_name, "override")

    def test_rejects_unknown_yaml_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_config = Path(temp_dir) / "unknown-key.yaml"
            bad_config.write_text("exp-name: test\nmodel: SiT-B/2\nmisspelled-key: 1\n")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--config", str(bad_config)])

    def test_rejects_invalid_yaml_choice_and_boolean(self):
        invalid_configs = (
            "exp-name: test\nmodel: SiT-B/2\nresolution: 300\n",
            'exp-name: test\nmodel: SiT-B/2\nfused-attn: "false"\n',
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, content in enumerate(invalid_configs):
                with self.subTest(content=content):
                    bad_config = Path(temp_dir) / f"invalid-{index}.yaml"
                    bad_config.write_text(content)
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            parse_args(["--config", str(bad_config)])


if __name__ == "__main__":
    unittest.main()
