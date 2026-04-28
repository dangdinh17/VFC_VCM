from collections import deque
from typing import Any, Deque, List, Optional, Union
import torch


class FeatureBuffer:
	"""A fixed-size FILO buffer for frame features.

	Internally uses a deque where the right side is the newest element.
	When pushing and the buffer is at capacity, the oldest (left-most)
	element is dropped so the most recent `max_size` elements are kept.

	Methods:
	- push(feature): add a feature to the buffer
	- pop(): remove and return the most recent feature (or None)
	- get(n): return up to `n` most recent features (newest first) without removing
	- peek(): return the most recent feature without removing
	- is_empty(), is_full(), reset(), all(), __len__()
	"""

	def __init__(self, max_size: int = 3) -> None:
		if max_size <= 0:
			raise ValueError("max_size must be >= 1")
		self.max_size: int = int(max_size)
		# right-most = newest
		self._data: Deque[Any] = deque()

	def push(self, feature: Any) -> None:
		"""Push a feature onto the buffer.

		If buffer is full, the oldest feature is discarded so the newest
		stays in the buffer (first-in, last-out semantics maintained).
		"""
		if len(self._data) >= self.max_size:
			# drop oldest (left-most)
			self._data.popleft()
		self._data.append(feature)

	def pop(self) -> Optional[Any]:
		"""Pop and return the most recently pushed feature, or None if empty."""
		if not self._data:
			return None
		return self._data.pop()

	def get(self, n=1) -> Union[List[Any], torch.Tensor]:
		if n <= 0:
			return None
		items = list(self._data)
		selected = list(reversed(items[-n:]))
		if selected:
			return torch.cat(selected, dim=1)
		return None
		
			
	def peek(self) -> Optional[Any]:
		"""Return the most recent feature without removing it, or None."""
		if not self._data:
			return None
		return self._data[-1]

	def reset(self) -> None:
		"""Remove all features from the buffer."""
		self._data.clear()

	def all(self) -> List[Any]:
		"""Return a shallow copy of all features as oldest->newest."""
		return list(self._data)

	def is_empty(self) -> bool:
		return len(self._data) == 0

	def is_full(self) -> bool:
		return len(self._data) >= self.max_size

	def __len__(self) -> int:
		return len(self._data)


__all__ = ["FeatureBuffer"]

