"""Helper functions to extract pose keypoints and skeleton renders.

This module is intentionally lightweight so other projects in this monorepo
can import it without depending on demo scripts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PoseResult:
	"""OpenPose body pose output."""
	candidate: np.ndarray
	subset: np.ndarray
	skeleton_bgr: np.ndarray


def create_body_estimator(*, model_dir: str | Path) -> Any:
	"""Create the OpenPose Body estimator.

	Args:
	  model_dir: directory containing `body_pose_model.pth`.
	"""
	from src.body import Body

	model_path = Path(model_dir) / "body_pose_model.pth"
	if not model_path.exists():
		raise FileNotFoundError(f"Missing body pose model: {model_path}")
	return Body(str(model_path))


def infer_body_pose(
	body_estimator: Any,
	bgr: np.ndarray,
	*,
	render: bool = True,
) -> PoseResult:
	"""Run body pose estimation.

	Args:
	  body_estimator: instance returned by `create_body_estimator()`.
	  bgr: input image as a numpy array (H, W, 3) in BGR order.
	  render: whether to also render the skeleton overlay.

	Returns:
	  PoseResult with candidate/subset arrays and a skeleton visualization image.
	"""
	if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
		raise ValueError(f"Expected BGR image HxWx3, got shape={getattr(bgr, 'shape', None)}")

	candidate, subset = body_estimator(bgr)
	# ensure numpy arrays (some versions return lists)
	candidate = np.asarray(candidate)
	subset = np.asarray(subset)

	if render:
		from src import util
		canvas = copy.deepcopy(bgr)
		canvas = util.draw_bodypose(canvas, candidate, subset)
	else:
		canvas = copy.deepcopy(bgr)

	return PoseResult(candidate=candidate, subset=subset, skeleton_bgr=canvas)


def pose_result_to_jsonable(result: PoseResult) -> Dict[str, Any]:
	"""Convert PoseResult to a JSON-serializable dict."""
	return {
		"candidate": result.candidate.tolist(),
		"subset": result.subset.tolist(),
		"candidate_shape": list(result.candidate.shape),
		"subset_shape": list(result.subset.shape),
	}


def save_skeleton_image_bgr(bgr: np.ndarray, out_path: str | Path) -> None:
	"""Save a BGR image to disk."""
	import cv2

	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	ok = cv2.imwrite(str(out_path), bgr)
	if not ok:
		raise IOError(f"Failed to write image: {out_path}")
