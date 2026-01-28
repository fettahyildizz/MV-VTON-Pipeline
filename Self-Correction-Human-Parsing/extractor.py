"""Parsing model extractor functions."""
import os
from collections import OrderedDict
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from utils.transforms import get_affine_transform
from utils.transforms import transform_logits

import networks

import argparse
from PIL import Image


dataset_settings = {
	'lip': {
		'input_size': [473, 473],
		'num_classes': 20,
		'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
				  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
				  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
	},
	'atr': {
		'input_size': [512, 512],
		'num_classes': 18,
		'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
				  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
	},
	'pascal': {
		'input_size': [512, 512],
		'num_classes': 7,
		'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
	}
}


def create_model(args: Any, *, arch: str = 'resnet101', device: Optional[torch.device] = None) -> torch.nn.Module:
	"""Create and load the parsing model based on args.

	Mirrors the model init/load behavior used in simple_extractor.py.

	Expected args fields:
	  - args.dataset: one of ['lip', 'atr', 'pascal']
	  - args.model_restore: path to checkpoint (optional)
	  - args.gpu: e.g. '0' or 'None' (optional)
	"""
	if getattr(args, 'gpu', None) is not None and args.gpu != 'None':
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	num_classes = dataset_settings[args.dataset]['num_classes']
	model = networks.init_model(arch, num_classes=num_classes, pretrained=None)

	model_restore = getattr(args, 'model_restore', '')
	if not model_restore:
		raise ValueError(
			"Missing --model-restore checkpoint path. "
			"Without pretrained weights the output will be incorrect."
		)

	checkpoint = torch.load(model_restore, map_location='cpu')
	state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		# simple_extractor.py strips the leading 'module.'
		name = k[7:] if k.startswith('module.') else k
		new_state_dict[name] = v
	model.load_state_dict(new_state_dict)

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	model.eval()
	return model


def get_model_output(
	model: torch.nn.Module,
	image: Union[torch.Tensor, "np.ndarray"],
	input_size: Union[Sequence[int], Tuple[int, int]],
	*,
	return_meta: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
	"""Run the model and return the upsampled final logits.

	Matches the forward/upsample used in simple_extractor.py:
	  output = model(image)
	  upsample(output[0][-1][0]) -> (H, W, C)

	Args:
	  model: parsing model (in eval mode)
	  image: either
		- a torch tensor shaped (N, C, H, W) or (C, H, W), OR
		- a raw OpenCV image (numpy array, HxWxC, BGR, uint8)
	  input_size: (height, width)
	  return_meta: when True and image is a cv2 numpy image, also returns the meta dict
				   (center/scale/width/height/rotation) compatible with transform_logits.

	Returns:
	  logits tensor of shape (H, W, C) on CPU, or (logits, meta) if return_meta=True.
	"""
	meta: Optional[dict] = None
	input_h, input_w = int(input_size[0]), int(input_size[1])

	# If a raw cv2 image is passed, preprocess it the same way as SimpleFolderDataset.
	if isinstance(image, np.ndarray):
		import cv2
		if image.ndim != 3 or image.shape[2] != 3:
			raise ValueError(f"Expected cv2 image HxWx3, got shape={image.shape}")
		h, w, _ = image.shape

		aspect_ratio = input_w * 1.0 / input_h
		center = np.zeros((2,), dtype=np.float32)
		center[0] = (w - 1) * 0.5
		center[1] = (h - 1) * 0.5
		box_w, box_h = float(w - 1), float(h - 1)
		if box_w > aspect_ratio * box_h:
			box_h = box_w / aspect_ratio
		elif box_w < aspect_ratio * box_h:
			box_w = box_h * aspect_ratio
		scale = np.array([box_w, box_h], dtype=np.float32)
		rotation = 0

		trans = get_affine_transform(center, scale, rotation, np.asarray([input_h, input_w]))
		warped = cv2.warpAffine(
			image,
			trans,
			(input_w, input_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)

		# Equivalent to torchvision: ToTensor() + Normalize(mean=[0.406,0.456,0.485], std=[0.225,0.224,0.229])
		# Note: those mean/std values are in BGR order, matching cv2.imread() behavior.
		tensor = torch.from_numpy(warped.astype(np.float32) / 255.0).permute(2, 0, 1)
		mean = torch.tensor([0.406, 0.456, 0.485], dtype=tensor.dtype).view(3, 1, 1)
		std = torch.tensor([0.225, 0.224, 0.229], dtype=tensor.dtype).view(3, 1, 1)
		tensor = (tensor - mean) / std
		image = tensor.unsqueeze(0)

		meta = {
			'center': center,
			'scale': scale,
			'width': w,
			'height': h,
			'rotation': rotation,
		}
	elif isinstance(image, torch.Tensor):
		if image.dim() == 3:
			image = image.unsqueeze(0)
	else:
		raise TypeError(f"Unsupported image type: {type(image)}")

	device = next(model.parameters()).device
	image = image.to(device)

	with torch.no_grad():
		output = model(image)
		upsample = torch.nn.Upsample(size=tuple(input_size), mode='bilinear', align_corners=True)
		upsample_output = upsample(output[0][-1][0].unsqueeze(0))
		upsample_output = upsample_output.squeeze(0).permute(1, 2, 0).contiguous()  # CHW -> HWC

	logits = upsample_output.cpu()
	if return_meta:
		return logits, (meta if meta is not None else {})
	return logits


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def _get_arguments():
	parser = argparse.ArgumentParser(description="Self Correction for Human Parsing (single image)")
	parser.add_argument("--dataset", type=str, default='atr', choices=['lip', 'atr', 'pascal'])
	parser.add_argument("--model-restore", type=str, default='models/atr-schp-201908301523-atr.pth', help="restore pretrained model parameters.")
	parser.add_argument("--gpu", type=str, default='0', help="choose gpu device (e.g. '0' or 'None').")
	parser.add_argument("--input-path", type=str, required=True, help="path of input image file.")
	parser.add_argument("--output-dir", type=str, default='./outputs', help="path of output folder.")
	parser.add_argument("--logits", action='store_true', default=False, help="whether to also save logits as .npy.")
	return parser.parse_args()


def main():
	args = _get_arguments()

	input_size = dataset_settings[args.dataset]['input_size']
	num_classes = dataset_settings[args.dataset]['num_classes']

	import cv2
	bgr = cv2.imread(args.input_path, cv2.IMREAD_COLOR)
	if bgr is None:
		raise FileNotFoundError(f"Could not read image: {args.input_path}")

	model = create_model(args)
	logits_hwc, meta = get_model_output(model, bgr, input_size, return_meta=True)

	c = meta['center']
	s = meta['scale']
	w = meta['width']
	h = meta['height']
	logits_result = transform_logits(logits_hwc.numpy(), c, s, w, h, input_size=input_size)
	parsing_result = np.argmax(logits_result, axis=2)

	os.makedirs(args.output_dir, exist_ok=True)
	base = os.path.splitext(os.path.basename(args.input_path))[0]
	out_png = os.path.join(args.output_dir, base + '.png')

	palette = get_palette(num_classes)
	output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
	output_img.putpalette(palette)
	output_img.save(out_png)

	if args.logits:
		out_npy = os.path.join(args.output_dir, base + '.npy')
		np.save(out_npy, logits_result)


if __name__ == '__main__':
	main()

