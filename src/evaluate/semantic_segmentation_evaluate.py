import torch

class EvaluatorTorch:
	def __init__(self, num_class: int) -> None:
		self.num_class = int(num_class)
		self.confusion_matrix = torch.zeros((self.num_class, self.num_class), dtype=torch.float64)

	def reset(self) -> None:
		self.confusion_matrix.zero_()

	def beforeval(self) -> None:
		isval = self.confusion_matrix.sum(dim=1) > 0
		self.confusion_matrix = self.confusion_matrix * isval[:, None]

	def pixel_accuracy(self) -> float:
		total = self.confusion_matrix.sum()
		if total <= 0:
			return 0.0
		return (torch.diag(self.confusion_matrix).sum() / total).item()

	def pixel_accuracy_class(self) -> float:
		denom = self.confusion_matrix.sum(dim=1)
		acc = torch.diag(self.confusion_matrix) / torch.clamp(denom, min=1.0)
		isval = denom > 0
		if isval.any():
			return acc[isval].mean().item()
		return 0.0

	def mean_iou(self) -> float:
		inter = torch.diag(self.confusion_matrix)
		union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - inter
		iou = inter / torch.clamp(union, min=1.0)
		isval = self.confusion_matrix.sum(dim=1) > 0
		if isval.any():
			return iou[isval].mean().item()
		return 0.0

	def fw_iou(self) -> float:
		total = self.confusion_matrix.sum()
		if total <= 0:
			return 0.0
		freq = self.confusion_matrix.sum(dim=1) / total
		inter = torch.diag(self.confusion_matrix)
		union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - inter
		iu = inter / torch.clamp(union, min=1.0)
		valid = freq > 0
		return (freq[valid] * iu[valid]).sum().item()

	def add_batch(self, gt_image: torch.Tensor, pred_image: torch.Tensor, ignore_index: int = 255) -> None:
		if gt_image.shape != pred_image.shape:
			raise ValueError(f"Shape mismatch: gt={gt_image.shape}, pred={pred_image.shape}")
		gt = gt_image.reshape(-1).to(torch.int64)
		pred = pred_image.reshape(-1).to(torch.int64)
		mask = (gt >= 0) & (gt < self.num_class) & (gt != ignore_index)
		if not mask.any():
			return
		label = self.num_class * gt[mask] + pred[mask]
		count = torch.bincount(label, minlength=self.num_class * self.num_class).to(torch.float64)
		self.confusion_matrix += count.reshape(self.num_class, self.num_class)

