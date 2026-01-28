"""MV-VTON inference script.

This script mirrors the sampling logic in `test.py`, but adds a more robust
dataset import so it can run even if `ldm/data/cp_dataset.py` is not present.

Expected data layout (paired):
  <dataroot>/<mode>/
	image-wo-bg/<folder>/{...3 views...}
	cloth/<folder>/{front, person, back}
	cloth-mask/<folder>/{front, ..., back}
	inpaint_mask/<folder>/{person-view mask}
	skeletons/<folder>/{front, person, back}
	warp_feat/<folder>.jpg

Unpaired additionally needs:
  <dataroot>/unpaired.txt
  <dataroot>/<mode>/warp_feat_unpair/<folder>.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
import hashlib
import types
import gc
from pathlib import Path

import numpy as np
import torch
import diffusers
from contextlib import nullcontext
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import Resize
from einops import rearrange
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

MONOREPO_ROOT = PROJECT_ROOT.parent

# In-process cache (best-effort). MV-VTON still reads from disk via CPDataset,
# and in single-pair mode we can avoid disk entirely.
_MEM_CACHE: dict[str, dict] = {}

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def _patch_lightning_legacy_imports() -> None:
	"""Patch missing legacy Lightning module paths used by older checkpoints.

	Some PyTorch Lightning 1.x checkpoints reference modules that were removed or
	moved in Lightning 2.x (e.g. `pytorch_lightning.utilities.distributed`).
	When loading such a checkpoint with `torch.load(..., weights_only=False)`,
	Python import during unpickling can fail with ModuleNotFoundError.

	This shim inserts a minimal alias module into `sys.modules` so unpickling can
	resolve common symbols.
	"""
	name = "pytorch_lightning.utilities.distributed"
	if name in sys.modules:
		return
	try:
		import pytorch_lightning  # noqa: F401
		from pytorch_lightning.utilities.rank_zero import (
			rank_zero_only,
			rank_zero_info,
			rank_zero_warn,
		)
	except Exception:
		# If Lightning isn't importable, nothing to patch.
		return

	mod = types.ModuleType(name)
	mod.rank_zero_only = rank_zero_only  # type: ignore[attr-defined]
	mod.rank_zero_info = rank_zero_info  # type: ignore[attr-defined]
	mod.rank_zero_warn = rank_zero_warn  # type: ignore[attr-defined]
	sys.modules[name] = mod


# def _check_runtime_deps() -> None:
# 	"""Fail fast with a helpful message when the environment is incompatible."""
# 	try:
# 		import diffusers  # noqa: F401
# 	except Exception as e:
# 		raise RuntimeError(
# 			"Missing dependency: `diffusers` is not installed in your current Python environment.\n"
# 			"Install it (plus compatible `transformers`/`huggingface-hub`) in the same env where you run inference, then retry."
# 		) from e


def _resolve_path(p: str) -> str:
	"""Resolve a path relative to CWD or this file's directory."""
	if not p:
		return p
	candidate = Path(p)
	if candidate.exists():
		return str(candidate)
	candidate = (PROJECT_ROOT / p).resolve()
	return str(candidate)


def _sha256_file(path: str | Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def _mask2bbox_deterministic(mask: np.ndarray, *, expand: float = 0.15) -> tuple[int, int, int, int]:
	"""Deterministic version of CPDataset.mask2bbox().

	Args:
	  mask: HxW boolean/0-1 mask
	  expand: bbox expansion factor around the center
	"""
	ys, xs = np.where(mask)
	if len(ys) == 0 or len(xs) == 0:
		# fallback to full image
		h, w = mask.shape[:2]
		return (0, h, 0, w)

	up = int(np.max(ys))
	down = int(np.min(ys))
	left = int(np.min(xs))
	right = int(np.max(xs))
	center_y = (up + down) // 2
	center_x = (left + right) // 2

	factor = float(expand)
	h, w = mask.shape[:2]
	up2 = int(min(up * (1 + factor) - center_y * factor + 1, h))
	down2 = int(max(down * (1 + factor) - center_y * factor, 0))
	left2 = int(max(left * (1 + factor) - center_x * factor, 0))
	right2 = int(min(right * (1 + factor) - center_x * factor + 1, w))
	return (down2, up2, left2, right2)


def _mask2bbox_like_dataset(
	mask: np.ndarray,
	*,
	seed: int | None = None,
) -> tuple[int, int, int, int]:
	"""Match CPDataset.mask2bbox() behavior, with optional deterministic seeding.

	CPDataset expands the bbox by a random factor in [0.1, 0.2). For single-pair
	mode we reproduce that expansion but allow a stable seed so results are
	repeatable across runs.
	"""
	import random as _random

	ys, xs = np.where(mask)
	if len(ys) == 0 or len(xs) == 0:
		h, w = mask.shape[:2]
		return (0, h, 0, w)

	up = int(np.max(ys))
	down = int(np.min(ys))
	left = int(np.min(xs))
	right = int(np.max(xs))
	center_y = (up + down) // 2
	center_x = (left + right) // 2

	rng = _random.Random(seed) if seed is not None else _random
	factor = rng.random() * 0.1 + 0.1

	h, w = mask.shape[:2]
	up2 = int(min(up * (1 + factor) - center_y * factor + 1, h))
	down2 = int(max(down * (1 + factor) - center_y * factor, 0))
	left2 = int(max(left * (1 + factor) - center_x * factor, 0))
	right2 = int(min(right * (1 + factor) - center_x * factor + 1, w))
	return (down2, up2, left2, right2)


def _derive_cloth_mask(cloth_rgb_path: Path, out_mask_path: Path) -> None:
	"""Derive a binary cloth mask from the cloth image.

	Heuristic:
	- If the image has alpha, use it.
	- Else treat near-white background as background.
	"""
	from PIL import Image
	import numpy as _np

	out_mask_path.parent.mkdir(parents=True, exist_ok=True)
	img = Image.open(cloth_rgb_path)
	if img.mode in ("RGBA", "LA"):
		alpha = _np.array(img.split()[-1])
		mask = (alpha > 0).astype(_np.uint8) * 255
	else:
		rgb = _np.array(img.convert("RGB"))
		# background if all channels are high
		bg = (rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245)
		mask = (~bg).astype(_np.uint8) * 255

	Image.fromarray(mask, mode="L").save(out_mask_path)


def _run_schp_parsing(
	person_bgr: np.ndarray,
	*,
	dataset: str,
	model_restore: str,
	gpu: str,
) -> tuple[np.ndarray, dict]:
	"""Run SCHP parsing and return (parsing_hw, meta)."""
	# Import from the sibling project.
	schp_root = MONOREPO_ROOT / "Self-Correction-Human-Parsing"
	if str(schp_root) not in sys.path:
		sys.path.insert(0, str(schp_root))

	from extractor import create_model as _create_model  # type: ignore
	from extractor import dataset_settings as _dataset_settings  # type: ignore
	from extractor import get_model_output as _get_model_output  # type: ignore
	from utils.transforms import transform_logits as _transform_logits  # type: ignore

	class _Args:
		pass

	args = _Args()
	args.dataset = dataset
	args.model_restore = model_restore
	args.gpu = gpu

	def _wants_cpu(g: str) -> bool:
		g2 = (g or "").strip().lower()
		return g2 in {"none", "cpu", "-1"}

	# Important: SCHP's create_model() defaults to CUDA if torch.cuda.is_available(),
	# which can crash on older GPUs with "no kernel image is available".
	# We explicitly control the device here.
	if _wants_cpu(gpu):
		device = torch.device("cpu")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	input_size = _dataset_settings[dataset]["input_size"]
	model = None
	try:
		model = _create_model(args, device=device)
		try:
			logits_hwc, meta = _get_model_output(model, person_bgr, input_size, return_meta=True)
		except RuntimeError as e:
			msg = str(e)
			# Common when torch was built without support for the GPU's compute capability.
			if ("no kernel image is available" in msg) or ("CUDA error" in msg and "no kernel image" in msg):
				print(
					"[SCHP] CUDA kernel image unavailable for this GPU; falling back to CPU for parsing.",
					file=sys.stderr,
				)
				# Recreate on CPU and rerun.
				del model
				model = _create_model(args, device=torch.device("cpu"))
				logits_hwc, meta = _get_model_output(model, person_bgr, input_size, return_meta=True)
			else:
				raise
	finally:
		# Best-effort cleanup: these models are only needed for preprocessing.
		try:
			del model
		except Exception:
			pass
		gc.collect()
		if torch.cuda.is_available():
			try:
				torch.cuda.empty_cache()
			except Exception:
				pass
	logits_result = _transform_logits(
		logits_hwc.numpy(),
		meta["center"],
		meta["scale"],
		meta["width"],
		meta["height"],
		input_size=input_size,
	)
	parsing = np.argmax(logits_result, axis=2).astype(np.uint8)
	return parsing, {"input_size": input_size, **meta}


def _parsing_to_inpaint_mask(parsing_hw: np.ndarray, *, dataset: str) -> np.ndarray:
	"""Create an inpaint mask from parsing labels.

	Return a uint8 mask where 255 indicates region to inpaint.
	"""
	if dataset == "atr":
		# ATR labels: 4=Upper-clothes, 7=Dress, 17=Scarf
		region = (parsing_hw == 4) | (parsing_hw == 7) | (parsing_hw == 17)
	elif dataset == "lip":
		# LIP labels: 5=Upper-clothes, 6=Dress, 7=Coat, 11=Scarf
		region = (parsing_hw == 5) | (parsing_hw == 6) | (parsing_hw == 7) | (parsing_hw == 11)
	else:
		# Fallback: torso-ish (class 2 in pascal settings here)
		region = parsing_hw != 0

	return (region.astype(np.uint8) * 255)


def _make_warp_feat_and_mask(person_rgb: np.ndarray, inpaint_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Create warp_feat (inpaint image) and inpaint mask image.

	- warp_feat: person image with inpaint region filled with 128 gray.
	- mask: grayscale 0/255 where 255 marks region to inpaint.
	"""
	warp = person_rgb.copy()
	if inpaint_mask.ndim != 2:
		raise ValueError("inpaint_mask must be HxW")
	region = inpaint_mask > 0
	warp[region] = np.array([128, 128, 128], dtype=np.uint8)
	return warp, inpaint_mask


def _run_openpose_skeleton(person_bgr: np.ndarray, *, model_dir: str) -> tuple[np.ndarray, dict]:
	openpose_root = MONOREPO_ROOT / "pytorch-openpose"
	if str(openpose_root) not in sys.path:
		sys.path.insert(0, str(openpose_root))

	from extract_keypoints import create_body_estimator, infer_body_pose, pose_result_to_jsonable  # type: ignore

	body = None
	result = None
	try:
		body = create_body_estimator(model_dir=model_dir)
		result = infer_body_pose(body, person_bgr, render=True)
		meta = pose_result_to_jsonable(result)
		return result.skeleton_bgr, meta
	finally:
		# Best-effort cleanup (OpenPose can be GPU-backed depending on install).
		try:
			del result
		except Exception:
			pass
		try:
			del body
		except Exception:
			pass
		gc.collect()
		if torch.cuda.is_available():
			try:
				torch.cuda.empty_cache()
			except Exception:
				pass


class _SinglePairDataset(torch.utils.data.Dataset):
	"""In-memory dataset producing a single MV-VTON sample."""

	def __init__(
		self,
		*,
		folder: str,
		image_size: int,
		person_rgb: np.ndarray,
		cloth_rgb: np.ndarray,
		cloth_mask_u8: np.ndarray,
		warp_feat_rgb: np.ndarray,
		inpaint_mask_u8: np.ndarray,
		skeleton_rgb: np.ndarray,
		ref_crop_seed: int | None = None,
	):
		super().__init__()
		self.folders = [folder]
		self._folder = folder
		self._order = folder.split("_")[1] if "_" in folder else "1"
		self._crop_size = (int(image_size), int(image_size / 256 * 256))
		self._ref_crop_seed = ref_crop_seed

		# Transforms match CPDataset behavior.
		import torchvision.transforms as _T
		self._transform = _T.Compose([
			_T.ToTensor(),
			_T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		self._transform_mask = _T.ToTensor()
		self._clip_normalize = _T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

		from PIL import Image as _Image
		# Store as PIL for consistent resizing behavior.
		self._person_pil = _Image.fromarray(person_rgb.astype(np.uint8), mode="RGB")
		self._cloth_pil = _Image.fromarray(cloth_rgb.astype(np.uint8), mode="RGB")
		self._cloth_mask_pil = _Image.fromarray(cloth_mask_u8.astype(np.uint8), mode="L")
		self._warp_feat_pil = _Image.fromarray(warp_feat_rgb.astype(np.uint8), mode="RGB")
		self._inpaint_mask_pil = _Image.fromarray(inpaint_mask_u8.astype(np.uint8), mode="L")
		self._skeleton_pil = _Image.fromarray(skeleton_rgb.astype(np.uint8), mode="RGB")

	def __len__(self) -> int:
		return 1

	def __getitem__(self, index: int) -> dict:
		import torchvision.transforms as _T
		from PIL import Image as _Image
		import numpy as _np
		import torch as _torch

		resize_rgb = _T.Resize(self._crop_size, interpolation=_T.InterpolationMode.BILINEAR)
		# Match CPDataset: inpaint mask uses interpolation=2 (bilinear) even though it's a mask.
		resize_inpaint_mask = _T.Resize(self._crop_size, interpolation=_T.InterpolationMode.BILINEAR)
		# Match CPDataset: cloth mask uses nearest-neighbor.
		resize_cloth_mask = _T.Resize(self._crop_size, interpolation=_T.InterpolationMode.NEAREST)

		# Person image
		im_pil = resize_rgb(self._person_pil)
		im = self._transform(im_pil)

		# Warp feature (agnostic/inpaint image)
		inpaint = resize_rgb(self._warp_feat_pil)
		inpaint = self._transform(inpaint)

		# Inpaint mask (file semantics: 255=inpaint region). Dataset returns keep-mask via 1 - mask.
		inpaint_mask = resize_inpaint_mask(self._inpaint_mask_pil)
		inpaint_mask = self._transform_mask(inpaint_mask)

		# Skeletons (use same skeleton for cf/cb/p in this single-view fallback)
		skeleton = resize_rgb(self._skeleton_pil)
		skeleton = self._transform(skeleton)

		# Cloth + mask
		c = resize_rgb(self._cloth_pil)
		c_t = self._transform(c)
		controlnet_cond_f = c_t
		controlnet_cond_b = c_t

		cm = resize_cloth_mask(self._cloth_mask_pil)
		cm_array = (_np.array(cm) >= 128).astype(_np.float32)
		cm_t = _torch.from_numpy(cm_array).unsqueeze(0)

		seed_base = int(self._ref_crop_seed) if self._ref_crop_seed is not None else None
		down_f, up_f, left_f, right_f = _mask2bbox_like_dataset(cm_array > 0, seed=seed_base)
		down_b, up_b, left_b, right_b = _mask2bbox_like_dataset(cm_array > 0, seed=(None if seed_base is None else seed_base + 1))

		ref_crop_f = c_t[:, down_f:up_f, left_f:right_f]
		ref_crop_f = (ref_crop_f + 1.0) / 2.0
		ref_crop_f = _T.Resize((224, 224))(ref_crop_f)
		ref_crop_f = self._clip_normalize(ref_crop_f)

		ref_crop_b = c_t[:, down_b:up_b, left_b:right_b]
		ref_crop_b = (ref_crop_b + 1.0) / 2.0
		ref_crop_b = _T.Resize((224, 224))(ref_crop_b)
		ref_crop_b = self._clip_normalize(ref_crop_b)

		result = {
			"GT": im,
			"inpaint_image": inpaint,
			"inpaint_mask": 1.0 - inpaint_mask,
			"ref_imgs_f": ref_crop_f,
			"ref_imgs_b": ref_crop_b,
			"warp_feat": inpaint,
			"skeleton_cf": skeleton,
			"skeleton_cb": skeleton,
			"skeleton_p": skeleton,
			"order": self._order,
			"controlnet_cond_f": controlnet_cond_f,
			"controlnet_cond_b": controlnet_cond_b,
			"file_name": f"{self._folder}.jpg",
		}
		return result



def _prepare_single_pair_in_memory(
	*,
	person_image: str,
	cloth_image: str,
	cloth_mask: str | None,
	sample_folder: str,
	schp_dataset: str,
	schp_ckpt: str,
	schp_gpu: str,
	openpose_model_dir: str,
	image_size: int,
	output_dir: str | None,
	reuse_mem_cache: bool,
) -> tuple[torch.utils.data.Dataset, str]:
	"""Prepare a single MV-VTON sample entirely in memory.

	Returns:
	  (dataset, sample_key)
	"""
	person_path = Path(person_image)
	cloth_path = Path(cloth_image)
	if not person_path.exists():
		raise FileNotFoundError(f"person image not found: {person_path}")
	if not cloth_path.exists():
		raise FileNotFoundError(f"cloth image not found: {cloth_path}")

	person_sha = _sha256_file(person_path)
	cloth_sha = _sha256_file(cloth_path)
	sample_key = hashlib.sha256(f"{person_sha}:{cloth_sha}:{schp_dataset}".encode("utf-8")).hexdigest()
	# Stable seed used to reproduce CPDataset's randomized reference crop.
	ref_crop_seed = int(sample_key[:8], 16)
	if reuse_mem_cache and sample_key in _MEM_CACHE:
		cached = _MEM_CACHE[sample_key]
		ds = cached["dataset"]
		return ds, sample_key

	import cv2
	from PIL import Image as _Image

	person_bgr = cv2.imread(str(person_path), cv2.IMREAD_COLOR)
	if person_bgr is None:
		raise FileNotFoundError(f"Could not read person image: {person_path}")
	person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)

	cloth_bgr = cv2.imread(str(cloth_path), cv2.IMREAD_COLOR)
	if cloth_bgr is None:
		raise FileNotFoundError(f"Could not read cloth image: {cloth_path}")
	cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB)

	# Cloth mask (optional)
	if cloth_mask is not None and cloth_mask != "":
		mask_src = Path(cloth_mask)
		if not mask_src.exists():
			raise FileNotFoundError(f"cloth mask not found: {mask_src}")
		mask_img = _Image.open(mask_src).convert("L")
		cloth_mask_u8 = np.array(mask_img, dtype=np.uint8)
	else:
		# Derive from the cloth RGB image directly.
		# background if all channels are high
		bg = (cloth_rgb[:, :, 0] > 245) & (cloth_rgb[:, :, 1] > 245) & (cloth_rgb[:, :, 2] > 245)
		cloth_mask_u8 = (~bg).astype(np.uint8) * 255

	parsing_hw, parsing_meta = _run_schp_parsing(
		person_bgr,
		dataset=schp_dataset,
		model_restore=schp_ckpt,
		gpu=schp_gpu,
	)

	# Optionally persist intermediate visualizations for debugging/inspection.
	if output_dir:
		try:
			out_base = Path(output_dir) / "intermediates" / sample_folder
			out_base.mkdir(parents=True, exist_ok=True)
			# Save a colorized parsing PNG (with palette) and a raw label-map PNG.
			schp_root = MONOREPO_ROOT / "Self-Correction-Human-Parsing"
			if str(schp_root) not in sys.path:
				sys.path.insert(0, str(schp_root))
			from extractor import get_palette as _get_palette  # type: ignore
			from extractor import dataset_settings as _ds  # type: ignore
			palette = _get_palette(_ds[schp_dataset]["num_classes"])
			parsing_img = _Image.fromarray(np.asarray(parsing_hw, dtype=np.uint8))
			parsing_img.putpalette(palette)
			parsing_img.save(out_base / "schp_parsing.png")
			_Image.fromarray(np.asarray(parsing_hw, dtype=np.uint8), mode="L").save(out_base / "schp_parsing_raw.png")
		except Exception as e:
			print(f"[warn] Failed to write SCHP parsing images: {e}", file=sys.stderr)
	inpaint_mask_u8 = _parsing_to_inpaint_mask(parsing_hw, dataset=schp_dataset)
	warp_rgb, _ = _make_warp_feat_and_mask(person_rgb, inpaint_mask_u8)

	skel_bgr, pose_meta = _run_openpose_skeleton(person_bgr, model_dir=openpose_model_dir)
	skeleton_rgb = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2RGB)
	if output_dir:
		try:
			out_base = Path(output_dir) / "intermediates" / sample_folder
			out_base.mkdir(parents=True, exist_ok=True)
			_Image.fromarray(np.asarray(skeleton_rgb, dtype=np.uint8), mode="RGB").save(out_base / "openpose_skeleton.png")
			_Image.fromarray(np.asarray(inpaint_mask_u8, dtype=np.uint8), mode="L").save(out_base / "inpaint_mask.png")
		except Exception as e:
			print(f"[warn] Failed to write OpenPose/inpaint images: {e}", file=sys.stderr)

	dataset = _SinglePairDataset(
		folder=sample_folder,
		image_size=image_size,
		person_rgb=person_rgb,
		cloth_rgb=cloth_rgb,
		cloth_mask_u8=cloth_mask_u8,
		warp_feat_rgb=warp_rgb,
		inpaint_mask_u8=inpaint_mask_u8,
		skeleton_rgb=skeleton_rgb,
		ref_crop_seed=ref_crop_seed,
	)

	_MEM_CACHE[sample_key] = {
		"dataset": dataset,
		"meta": {"schp": parsing_meta, "openpose": pose_meta, "person_path": str(person_path), "cloth_path": str(cloth_path)},
	}
	return dataset, sample_key


def _import_cp_dataset(unpaired: bool):
	"""Import CPDataset.

	The repo convention is to symlink one of
	`cp_dataset_mv_{paired,unpaired}.py` -> `cp_dataset.py`.
	This fallback allows running without that symlink.
	"""
	try:
		from ldm.data.cp_dataset import CPDataset  # type: ignore
		return CPDataset
	except Exception:
		if unpaired:
			from ldm.data.cp_dataset_mv_unpaired import CPDataset  # type: ignore
		else:
			from ldm.data.cp_dataset_mv_paired import CPDataset  # type: ignore
		return CPDataset


def load_model_from_config(config, ckpt: str, verbose: bool = False):
	ckpt = _resolve_path(ckpt)
	print(f"Loading model from {ckpt}")
	# Newer PyTorch versions may default to a restricted "weights-only" unpickler
	# which rejects Lightning callback objects embedded in .ckpt files.
	# We only load checkpoints we trust here; disable weights-only restrictions.
	try:
		# _patch_lightning_legacy_imports()
		pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
	except TypeError:
		# Older torch doesn't support the weights_only kwarg.
		# _patch_lightning_legacy_imports()
		pl_sd = torch.load(ckpt, map_location="cpu")
	except ModuleNotFoundError as e:
		# Retry once after applying legacy Lightning module shims.
		# _patch_lightning_legacy_imports()
		pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
	if isinstance(pl_sd, dict) and "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd
	model = instantiate_from_config(config.model)
	missing, unexpected = model.load_state_dict(sd, strict=False)
	if verbose and missing:
		print("missing keys:")
		print(missing)
	if verbose and unexpected:
		print("unexpected keys:")
		print(unexpected)
	return model


def _to_device(batch: dict, device: torch.device) -> dict:
	out = {}
	for k, v in batch.items():
		if torch.is_tensor(v):
			out[k] = v.to(device)
		else:
			out[k] = v
	return out


def _cast_floats(batch: dict, *, dtype: torch.dtype) -> dict:
	"""Cast floating tensors in a batch to the requested dtype."""
	out = {}
	for k, v in batch.items():
		if torch.is_tensor(v) and torch.is_floating_point(v):
			out[k] = v.to(dtype=dtype)
		else:
			out[k] = v
	return out


def run_inference(opt: argparse.Namespace) -> None:
	seed_everything(opt.seed)
	# _check_runtime_deps()

	device = torch.device(f"cuda:{opt.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	compute_dtype = torch.float16 if (opt.use_fp16 and device.type == "cuda") else torch.float32

	config_path = _resolve_path(opt.config)
	ckpt_path = _resolve_path(opt.ckpt)
	config = OmegaConf.load(config_path)

	model = load_model_from_config(config, ckpt_path, verbose=opt.verbose)
	model = model.to(device)
	model.eval()
	if opt.use_fp16:
		print("Using fp16 precision for model.")
		model.half()

	outdir = Path(opt.outdir)
	outdir.mkdir(parents=True, exist_ok=True)
	result_path = outdir / "upper_body"
	result_path.mkdir(parents=True, exist_ok=True)

	# Dataset selection
	if getattr(opt, "person_image", "") and getattr(opt, "cloth_image", ""):
		dataset, sample_key = _prepare_single_pair_in_memory(
			person_image=opt.person_image,
			cloth_image=opt.cloth_image,
			cloth_mask=(opt.cloth_mask if opt.cloth_mask else None),
			sample_folder=opt.single_folder,
			schp_dataset=opt.schp_dataset,
			schp_ckpt=_resolve_path(opt.schp_ckpt),
			schp_gpu=opt.schp_gpu,
			openpose_model_dir=_resolve_path(opt.openpose_model_dir),
			image_size=opt.image_size,
			output_dir=str(outdir),
			reuse_mem_cache=bool(getattr(opt, "reuse_mem_cache", False)),
		)
		print(f"[single-pair] sample_key={sample_key}")
		# In single-pair mode, force running the only element.
		opt.folder = opt.single_folder
	else:
		CPDataset = _import_cp_dataset(unpaired=opt.unpaired)
		dataset = CPDataset(opt.dataroot, opt.image_size, mode=opt.mode, unpaired=opt.unpaired)

	# Single-sample mode: select one element by folder name or index.
	if opt.folder:
		if not hasattr(dataset, "folders"):
			raise ValueError("Dataset does not expose `folders`; cannot use --folder")
		try:
			selected_idx = dataset.folders.index(opt.folder)
		except ValueError as e:
			raise ValueError(
				f"Folder '{opt.folder}' not found in dataset. "
				f"Expected a name like '00001_1' present under '{opt.dataroot}/{opt.mode}/image-wo-bg/'."
			) from e
		dataset = Subset(dataset, [selected_idx])
	elif opt.index is not None:
		if opt.index < 0:
			raise ValueError("--index must be >= 0")
		dataset = Subset(dataset, [opt.index])

	loader = DataLoader(
		dataset,
		batch_size=opt.n_samples,
		shuffle=False,
		num_workers=opt.num_workers,
		pin_memory=(device.type == "cuda"),
	)

	sampler = PLMSSampler(model) if opt.plms else DDIMSampler(model)

	start_code = None
	if opt.fixed_code and device.type == "cuda":
		# only used if caller wants deterministic x_T; test.py overrides x_T from warp_feat anyway
		start_code = torch.randn([opt.n_samples, opt.C, opt.image_size // opt.f, opt.image_size // opt.f], device=device)

	iterator = tqdm(loader, desc=f"{opt.mode} dataset", total=len(loader))
	precision_scope = autocast if opt.precision == "autocast" else nullcontext

	num_saved = 0
	# with torch.no_grad():
	# inference_mode is stronger than no_grad and can reduce memory usage.
	with torch.inference_mode():
		# Autocast reduces activation memory; additionally, --use_fp16 casts inputs to float16.
		with precision_scope("cuda", dtype=compute_dtype) if device.type == "cuda" and opt.precision == "autocast" else nullcontext():
			# many Lightning-style diffusion repos expose ema_scope(); keep parity with test.py
			ema_ctx = model.ema_scope() if hasattr(model, "ema_scope") else nullcontext()
			with ema_ctx:
				for batch in iterator:
					batch = _to_device(batch, device)
					# Reduce VRAM: cast all float tensors to fp16 when requested.
					if opt.use_fp16 and device.type == "cuda":
						batch = _cast_floats(batch, dtype=torch.float16)

					if getattr(opt, "debug", False) and num_saved == 0:
						def _summ(name: str, t: torch.Tensor) -> str:
							return f"{name}: shape={tuple(t.shape)} dtype={t.dtype} min={float(t.min()):.4f} max={float(t.max()):.4f}"
						print("[debug] batch keys:", sorted(batch.keys()))
						for k in [
							"inpaint_mask",
							"inpaint_image",
							"ref_imgs_f",
							"ref_imgs_b",
							"warp_feat",
							"GT",
							"controlnet_cond_f",
							"controlnet_cond_b",
							"skeleton_cf",
							"skeleton_cb",
							"skeleton_p",
						]:
							v = batch.get(k)
							if torch.is_tensor(v):
								print("[debug]", _summ(k, v))
						mask_v = batch.get("inpaint_mask")
						if torch.is_tensor(mask_v):
							if float(mask_v.min()) < -1e-3 or float(mask_v.max()) > 1.0 + 1e-3:
								print("[debug][warn] inpaint_mask outside [0,1]", file=sys.stderr)

					mask_tensor = batch["inpaint_mask"]
					inpaint_image = batch["inpaint_image"]
					ref_tensor_f = batch["ref_imgs_f"]
					ref_tensor_b = batch["ref_imgs_b"]
					skeleton_cf = batch["skeleton_cf"]
					skeleton_cb = batch["skeleton_cb"]
					skeleton_p = batch["skeleton_p"]
					order = batch["order"]
					feat_tensor = batch["warp_feat"]
					image_tensor = batch["GT"]
					controlnet_cond_f = batch["controlnet_cond_f"]
					controlnet_cond_b = batch["controlnet_cond_b"]

					# Choose which reference embedding per sample based on view order
					ref_tensor = ref_tensor_f.clone()
					for i in range(len(order)):
						# `order` is a string like "1"/"2"/"3" (from folder suffix)
						if order[i] == "1" or order[i] == "2":
							continue
						if order[i] == "3":
							ref_tensor[i] = ref_tensor_b[i]
							continue
						raise ValueError(f"Invalid order: {order[i]}")

					test_model_kwargs = {
						"inpaint_mask": mask_tensor,
						"inpaint_image": inpaint_image,
					}

					uc = None
					if opt.scale != 1.0:
						uc = model.learnable_vector
						uc = uc.repeat(ref_tensor.size(0), 1, 1)

					# Match conditioning dtype to the compute dtype.
					c = model.get_learned_conditioning(ref_tensor.to(dtype=compute_dtype))
					c = model.proj_out(c)

					z_inpaint = model.encode_first_stage(test_model_kwargs["inpaint_image"])
					z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
					test_model_kwargs["inpaint_image"] = z_inpaint
					test_model_kwargs["inpaint_mask"] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
						test_model_kwargs["inpaint_mask"]
					)

					warp_feat = model.encode_first_stage(feat_tensor)
					warp_feat = model.get_first_stage_encoding(warp_feat).detach()

					ts = torch.full((1,), 999, device=device, dtype=torch.long)
					start_code = model.q_sample(warp_feat, ts)

					# local_controlnet
					ehs_cf = model.pose_model(skeleton_cf)
					ehs_cb = model.pose_model(skeleton_cb)
					ehs_p = model.pose_model(skeleton_p)
					# Match test.py behavior: ehs_text is float32 by default.
					ehs_text = torch.zeros((c.shape[0], 1, 768), device=device)

					x_noisy = torch.cat(
						(start_code, test_model_kwargs["inpaint_image"], test_model_kwargs["inpaint_mask"]), dim=1
					)

					down_samples_f, mid_samples_f = model.local_controlnet(
						x_noisy,
						ts,
						encoder_hidden_states=ehs_text,
						controlnet_cond=controlnet_cond_f,
						ehs_c=ehs_cf,
						ehs_p=ehs_p,
					)
					down_samples_b, mid_samples_b = model.local_controlnet(
						x_noisy,
						ts,
						encoder_hidden_states=ehs_text,
						controlnet_cond=controlnet_cond_b,
						ehs_c=ehs_cb,
						ehs_p=ehs_p,
					)

					_ = mid_samples_f + mid_samples_b
					down_samples = tuple(
						torch.cat((down_samples_f[ds], down_samples_b[ds]), dim=1) for ds in range(len(down_samples_f))
					)

					# Use latent spatial size derived from z_inpaint/start_code to avoid H/W mismatches.
					latent_shape = [start_code.shape[1], start_code.shape[2], start_code.shape[3]]

					try:
						samples, _ = sampler.sample(
							S=opt.ddim_steps,
							conditioning=c,
							batch_size=opt.n_samples,
							shape=latent_shape,
							verbose=False,
							unconditional_guidance_scale=opt.scale,
							unconditional_conditioning=uc,
							eta=opt.ddim_eta,
							x_T=start_code,
							down_samples=down_samples,
							test_model_kwargs=test_model_kwargs,
						)
					except torch.cuda.OutOfMemoryError as e:
						if device.type == "cuda":
							try:
								torch.cuda.empty_cache()
							except Exception:
								pass
							msg = (
								"CUDA OOM during sampling. Suggestions:\n"
								"- Close other GPU processes (check `nvidia-smi`).\n"
								"- Use `--n_samples 1` (batch size) and reduce `--image_size` to 384 or 256.\n"
								"- Reduce steps: `--ddim_steps 20` or lower.\n"
								"- Try fp16 weights/tensors: add `--use_fp16`.\n"
								"- Or run on CPU: add `--force-cpu`."
							)
							raise torch.cuda.OutOfMemoryError(msg) from e
						raise

					x_samples = model.decode_first_stage(samples)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
					x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

					x_checked_image_torch = torch.from_numpy(x_samples).permute(0, 3, 1, 2)
					x_source = torch.clamp((image_tensor.cpu() + 1.0) / 2.0, min=0.0, max=1.0)
					mask_cpu = mask_tensor.cpu()
					x_result = x_checked_image_torch * (1 - mask_cpu) + mask_cpu * x_source

					# Match test.py default output aspect (H x (H/256*192)) unless overridden.
					if opt.save_h > 0 and opt.save_w > 0:
						resize = transforms.Resize((opt.save_h, opt.save_w))
					else:
						resize = transforms.Resize((opt.image_size, int(opt.image_size / 256 * 192)))

					if not opt.skip_save:
						for i, x_sample in enumerate(x_result):
							filename = batch.get("file_name", [f"{num_saved:06d}.jpg"])[i]
							stem = str(filename)
							if stem.lower().endswith((".jpg", ".jpeg", ".png")):
								stem = os.path.splitext(stem)[0]

							save_x = resize(x_sample)
							save_x = 255.0 * rearrange(save_x.numpy(), "c h w -> h w c")
							img = Image.fromarray(save_x.astype(np.uint8))
							img.save(result_path / f"{stem}.png")
							num_saved += 1
							if opt.max_images > 0 and num_saved >= opt.max_images:
								return


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
	parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt)")
	parser.add_argument("--dataroot", type=str, default="", help="Dataset root (ignored in single-pair mode)")
	parser.add_argument("--mode", type=str, default="test", choices=["train", "test"], help="Dataset split")

	# Single-pair mode: if provided, parsing/pose/masks are generated and kept in memory.
	parser.add_argument("--person-image", type=str, default="", help="Single person image path (enables single-pair mode)")
	parser.add_argument("--cloth-image", type=str, default="", help="Single cloth image path (enables single-pair mode)")
	parser.add_argument("--cloth-mask", type=str, default="", help="Optional binary mask for cloth (white=cloth)")
	parser.add_argument("--reuse-mem-cache", action="store_true", help="Reuse computed assets within the same Python process")
	parser.add_argument("--single-folder", type=str, default="single_1", help="Synthetic folder name (must contain '_')")

	# SCHP parsing options
	parser.add_argument("--schp-dataset", type=str, default="atr", choices=["atr", "lip", "pascal"], help="SCHP label set")
	parser.add_argument(
		"--schp-ckpt",
		type=str,
		default=str(MONOREPO_ROOT / "Self-Correction-Human-Parsing" / "models" / "atr-schp-201908301523-atr.pth"),
		help="SCHP checkpoint path",
	)
	parser.add_argument("--schp-gpu", type=str, default="0", help="GPU id string for SCHP (e.g. '0' or 'None')")

	# OpenPose options
	parser.add_argument(
		"--openpose-model-dir",
		type=str,
		default=str(MONOREPO_ROOT / "pytorch-openpose" / "model"),
		help="Directory containing body_pose_model.pth",
	)

	# Diffusers base model (Paint-by-Example) used to initialize ControlNet architecture
	parser.add_argument(
		"--pbe-model",
		type=str,
		default="",
		help="Paint-by-Example model id or local directory (overrides MVVTON_PBE_MODEL).",
	)
	parser.add_argument(
		"--pbe-ckpt",
		type=str,
		default="",
		help="Optional single-file Paint-by-Example checkpoint (.ckpt/.safetensors). Sets MVVTON_PBE_CKPT.",
	)
	parser.add_argument(
		"--pbe-config",
		type=str,
		default="",
		help="Optional original SD YAML config for --pbe-ckpt conversion. Sets MVVTON_PBE_CONFIG.",
	)
	parser.add_argument(
		"--hf-local-only",
		action="store_true",
		help="Force HuggingFace local-files-only loading (sets MVVTON_HF_LOCAL_ONLY=1).",
	)

	parser.add_argument("--outdir", type=str, default="outputs/infer", help="Directory to write results")
	parser.add_argument("--use_fp16", action="store_true", default=False, help="Use fp16 precision for model")
	parser.add_argument("--gpu_id", type=int, default=0, help="Which GPU to use")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")

	parser.add_argument("--image_size", type=int, default=512, help="Dataset resize (height); width follows dataset")
	parser.add_argument("--n_samples", type=int, default=1, help="Batch size")
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--max_images", type=int, default=0, help="Stop after saving N images (0 = all)")
	parser.add_argument(
		"--folder",
		type=str,
		default="",
		help="Run a single sample by folder name (e.g. '00001_1').",
	)
	parser.add_argument(
		"--index",
		type=int,
		default=None,
		help="Run a single sample by dataset index (0-based). Ignored if --folder is set.",
	)

	parser.add_argument("--ddim_steps", type=int, default=30)
	parser.add_argument("--ddim_eta", type=float, default=0.0)
	parser.add_argument("--plms", action="store_true")
	parser.add_argument("--fixed_code", action="store_true")
	parser.add_argument("--scale", type=float, default=1.0)
	parser.add_argument("--unpaired", action="store_true", help="Use unpaired dataset variant")

	parser.add_argument("--C", type=int, default=4, help="Latent channels")
	parser.add_argument("--f", type=int, default=8, help="Downsampling factor")

	parser.add_argument("--save_h", type=int, default=0, help="Resize saved output height (0 = no resize)")
	parser.add_argument("--save_w", type=int, default=0, help="Resize saved output width (0 = no resize)")
	parser.add_argument("--skip_save", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Print extra tensor shape/range info for the first batch.",
	)

	opt = parser.parse_args()

	# Propagate HF/diffusers model selection to the model code (ddpm.py).
	if opt.pbe_model:
		os.environ["MVVTON_PBE_MODEL"] = opt.pbe_model
	if opt.pbe_ckpt:
		os.environ["MVVTON_PBE_CKPT"] = opt.pbe_ckpt
	if opt.pbe_config:
		os.environ["MVVTON_PBE_CONFIG"] = opt.pbe_config
	if opt.hf_local_only:
		os.environ["MVVTON_HF_LOCAL_ONLY"] = "1"

	# CPDataset expects folder names like 00001_1 and uses suffix as view order.
	if opt.person_image and opt.cloth_image and "_" not in opt.single_folder:
		raise ValueError("--single-folder must contain '_' (e.g. 'single_1')")

	if (not opt.person_image or not opt.cloth_image) and not opt.dataroot:
		raise ValueError("Provide --dataroot, or use --person-image and --cloth-image for single-pair mode")
	run_inference(opt)


if __name__ == "__main__":
	main()
