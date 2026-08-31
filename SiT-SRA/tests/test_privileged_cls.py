import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dataset import HFImgLatentDataset
from dino_cls import DINOCLSExtractor
from loss import SRALoss, linear_cls_probability
from models.sit import CLSTokenEmbedder, MaskedAttention, SiT


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, x, t, y, ad, cls_condition, cls_present):
        self.calls.append({
            "cls_condition": cls_condition.detach().clone(),
            "cls_present": cls_present.detach().clone(),
            "grad_enabled": torch.is_grad_enabled(),
        })
        features = x.flatten(2).transpose(1, 2) * self.scale
        return x * self.scale, features, y


class ProbabilityScheduleTest(unittest.TestCase):
    def test_linear_decay_and_floor(self):
        self.assertEqual(linear_cls_probability(0, 1.0, 0.2, 100), 1.0)
        self.assertAlmostEqual(linear_cls_probability(50, 1.0, 0.2, 100), 0.6)
        self.assertEqual(linear_cls_probability(100, 1.0, 0.2, 100), 0.2)
        self.assertEqual(linear_cls_probability(1000, 1.0, 0.2, 100), 0.2)


class CLSTokenEmbedderTest(unittest.TestCase):
    def test_mask_selects_clean_or_null_per_sample(self):
        embedder = CLSTokenEmbedder(input_dim=4, hidden_size=5)
        cls_condition = torch.randn(3, 4)

        clean = embedder(cls_condition, 3, cls_condition.device, cls_condition.dtype)

        self.assertEqual(clean.shape, (3, 5))
        with self.assertRaisesRegex(ValueError, "requires cls_condition"):
            embedder(None, 3, cls_condition.device, cls_condition.dtype)

    def test_original_sit_forward_has_no_cls_slot(self):
        model = SiT(
            input_size=4,
            patch_size=2,
            hidden_size=32,
            decoder_hidden_size=32,
            depth=2,
            num_heads=4,
            cls_condition_dim=0,
            qk_norm=False,
            fused_attn=False,
        ).eval()
        images = torch.randn(2, 4, 4, 4)
        timesteps = torch.rand(2)
        labels = torch.tensor([1, 2])
        seen = {}

        def record_block_input(_module, inputs):
            seen["token_shape"] = inputs[0].shape

        hook = model.blocks[0].register_forward_pre_hook(record_block_input)
        try:
            output, features, output_labels = model(images, timesteps, labels, ad=1)
        finally:
            hook.remove()

        self.assertEqual(seen["token_shape"], (2, 4, 32))
        self.assertEqual(output.shape, images.shape)
        self.assertEqual(features.shape, (2, 4, 32))
        torch.testing.assert_close(output_labels, labels)

    def test_cls_training_weights_strictly_load_into_original_sit(self):
        """The sampler must be able to use plain SiT after dropping CLS projector."""
        common = dict(
            input_size=4,
            patch_size=2,
            hidden_size=32,
            decoder_hidden_size=32,
            depth=2,
            num_heads=4,
            qk_norm=False,
            fused_attn=False,
        )
        cls_training_model = SiT(cls_condition_dim=6, **common)
        original_sit = SiT(cls_condition_dim=0, **common)
        inference_state = {
            key: value
            for key, value in cls_training_model.state_dict().items()
            if not key.startswith("cls_token_embedder.")
        }

        original_sit.load_state_dict(inference_state, strict=True)

    def test_masked_cls_is_invisible_to_patch_attention(self):
        attention = MaskedAttention(
            dim=8, num_heads=2, qk_norm=False, fused_attn=False
        ).eval()
        cls = torch.randn(2, 1, 8)
        patches = torch.randn(2, 4, 8)
        key_mask = torch.zeros(2, 1, 1, 5)
        key_mask[:, :, :, 0] = float("-inf")

        without_cls = attention(patches)
        with_masked_cls = attention(torch.cat([cls, patches], dim=1), key_mask=key_mask)

        torch.testing.assert_close(with_masked_cls[:, 1:], without_cls)

    def test_tiny_sit_runs_privileged_alignment_end_to_end(self):
        student = SiT(
            input_size=4,
            patch_size=2,
            hidden_size=32,
            decoder_hidden_size=32,
            depth=2,
            num_heads=4,
            cls_condition_dim=6,
            qk_norm=False,
            fused_attn=False,
        ).train()
        teacher = deepcopy(student).eval().requires_grad_(False)
        criterion = SRALoss(block_out_s=1, block_out_t=2, loss_type="l2")

        flow_loss, align_loss, mask = criterion(
            student,
            torch.randn(3, 4, 4, 4),
            teacher,
            torch.tensor([1, 2, 3]),
            torch.randn(3, 6),
            student_cls_probability=0.5,
        )
        loss = flow_loss.mean() + 0.5 * align_loss.mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(mask.shape, (3,))
        self.assertEqual(mask.dtype, torch.bool)


class PrivilegedTeacherLossTest(unittest.TestCase):
    def test_teacher_is_always_cls_on_and_has_no_grad(self):
        images = torch.randn(4, 2, 2, 2)
        labels = torch.arange(4)
        cls_condition = torch.randn(4, 6)
        criterion = SRALoss(block_out_s=1, block_out_t=2, loss_type="l2")

        for probability, expected_student_mask in (
            (0.0, torch.zeros(4, dtype=torch.bool)),
            (1.0, torch.ones(4, dtype=torch.bool)),
        ):
            with self.subTest(probability=probability):
                student = RecordingModel()
                teacher = RecordingModel().eval().requires_grad_(False)
                flow_loss, align_loss, sampled_mask = criterion(
                    student,
                    images,
                    teacher,
                    labels,
                    cls_condition,
                    probability,
                )
                (flow_loss.mean() + align_loss.mean()).backward()

                torch.testing.assert_close(sampled_mask, expected_student_mask)
                torch.testing.assert_close(
                    student.calls[0]["cls_present"], expected_student_mask
                )
                self.assertTrue(teacher.calls[0]["cls_present"].all())
                torch.testing.assert_close(
                    student.calls[0]["cls_condition"], cls_condition
                )
                torch.testing.assert_close(
                    teacher.calls[0]["cls_condition"], cls_condition
                )
                self.assertTrue(student.calls[0]["grad_enabled"])
                self.assertFalse(teacher.calls[0]["grad_enabled"])
                self.assertIsNone(teacher.scale.grad)


class StubDataset:
    def __init__(self, rows, columns):
        self.rows = rows
        self.column_names = columns

    def __getitem__(self, idx):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)


class HFDatasetTest(unittest.TestCase):
    def test_loads_sit_image_and_latent_pair(self):
        image_dataset = StubDataset(
            [{"image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)), "label": 7}],
            ["image", "label"],
        )
        latent_dataset = StubDataset(
            [{"data": np.ones((8, 2, 2), dtype=np.float32)}],
            ["data"],
        )

        with patch("dataset.load_from_disk", side_effect=[image_dataset, latent_dataset]) as loader:
            dataset = HFImgLatentDataset(
                "sdvae-ft-mse-f8d4", "/dev/shm/data", split="train"
            )
            raw_image, latent, label = dataset[0]

        self.assertEqual(raw_image.shape, (3, 8, 8))
        self.assertEqual(raw_image.dtype, torch.uint8)
        self.assertEqual(latent.shape, (8, 2, 2))
        self.assertEqual(label.item(), 7)
        self.assertEqual(
            loader.call_args_list[0].args[0].rstrip("/"),
            "/dev/shm/data/imagenet-latents-images",
        )
        self.assertEqual(
            loader.call_args_list[1].args[0].rstrip("/"),
            "/dev/shm/data/imagenet-latents-sdvae-ft-mse-f8d4",
        )

    def test_normalizes_legacy_singleton_latent_axis(self):
        image_dataset = StubDataset(
            [{"image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)), "label": 7}],
            ["image", "label"],
        )
        latent_dataset = StubDataset(
            [{"data": np.ones((1, 8, 2, 2), dtype=np.float32)}],
            ["data"],
        )

        with patch("dataset.load_from_disk", side_effect=[image_dataset, latent_dataset]):
            _, latent, _ = HFImgLatentDataset("sdvae-ft-mse-f8d4", "/dev/shm/data")[0]

        self.assertEqual(latent.shape, (8, 2, 2))

    def test_rejects_mismatched_pair_lengths(self):
        image_dataset = StubDataset([], ["image", "label"])
        latent_dataset = StubDataset([{"data": np.zeros(1)}], ["data"])
        with patch("dataset.load_from_disk", side_effect=[image_dataset, latent_dataset]):
            with self.assertRaisesRegex(ValueError, "lengths differ"):
                HFImgLatentDataset("sdvae-ft-mse-f8d4", "/dev/shm/data")


class RecordingDINO(nn.Module):
    embed_dim = 6

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.last_input = None

    def forward_features(self, images):
        self.last_input = images.detach().clone()
        return {
            "x_norm_clstoken": torch.ones(
                images.shape[0], self.embed_dim, device=images.device
            )
        }


class DINOCLSExtractorTest(unittest.TestCase):
    def test_extracts_clean_cls_from_sit_raw_images(self):
        backbone = RecordingDINO()
        extractor = DINOCLSExtractor(backbone, resolution=256)
        raw_images = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8)

        cls_condition = extractor(raw_images)

        self.assertEqual(cls_condition.shape, (2, 6))
        self.assertEqual(backbone.last_input.shape, (2, 3, 224, 224))
        self.assertTrue(torch.isfinite(backbone.last_input).all())
        self.assertFalse(backbone.weight.requires_grad)
        self.assertFalse(cls_condition.requires_grad)


if __name__ == "__main__":
    unittest.main()
