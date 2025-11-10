import hashlib

import torch


class TaskLabeler:
    """Injects optional task metadata into model inputs."""

    def __init__(self, config, base_n_dims):
        self.config = config
        self.enabled = bool(config and getattr(config, "enabled", False))
        self.label_dim = (
            int(getattr(config, "dimension", 1) or 0) if self.enabled else 0
        )
        if self.label_dim < 0:
            raise ValueError("task_labeling.dimension must be non-negative")

        self.feature_dims = base_n_dims
        self.model_n_dims = base_n_dims + (self.label_dim if self.enabled else 0)

        if self.enabled and self.feature_dims <= 0:
            raise ValueError(
                "Base n_dims must be positive when task labeling is enabled"
            )

        self.mode = getattr(config, "mode", "auto") if self.enabled else "auto"
        self.manual_map = {}
        if self.enabled and getattr(config, "manual_map", None):
            for name, value in config.manual_map.items():
                self.manual_map[name] = self._coerce_vector(value)

    def _coerce_vector(self, value):
        if isinstance(value, (int, float)):
            vec = [float(value)]
        else:
            vec = [float(v) for v in value]
        if len(vec) != self.label_dim:
            raise ValueError(
                f"Manual map for task label must have {self.label_dim} entries"
            )
        return torch.tensor(vec, dtype=torch.float32)

    def _auto_vector(self, label_name):
        digest = hashlib.sha1(label_name.encode("utf-8")).digest()
        values = []
        for i in range(self.label_dim):
            byte = digest[i % len(digest)]
            values.append(2 * (byte / 255.0) - 1.0)
        return torch.tensor(values, dtype=torch.float32)

    def _label_vector(self, label_name, device):
        if not self.enabled:
            raise RuntimeError("TaskLabeler is disabled")
        if label_name in self.manual_map:
            vec = self.manual_map[label_name]
        else:
            if self.mode == "manual":
                raise KeyError(
                    f"No manual mapping provided for task label '{label_name}'"
                )
            vec = self._auto_vector(label_name)
        return vec.to(device)

    def augmentation_truncation(self, feature_truncation):
        if not self.enabled:
            return feature_truncation
        if feature_truncation > self.feature_dims:
            raise ValueError(
                "Curriculum requested more feature dims than available"
            )
        return feature_truncation + self.label_dim

    def feature_slice(self, xs):
        if not self.enabled:
            return xs
        return xs[:, :, : self.feature_dims]

    def apply(self, xs, label_name):
        if not self.enabled or self.label_dim == 0:
            return xs
        bsz, n_points, _ = xs.shape
        label_tensor = self._label_vector(label_name, xs.device)
        label_tensor = label_tensor.view(1, 1, -1).expand(bsz, n_points, -1)
        xs[:, :, self.feature_dims :] = label_tensor
        return xs
