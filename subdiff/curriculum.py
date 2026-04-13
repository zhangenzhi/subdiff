"""
Curriculum learning scheduler for SubDiff pretraining.
Controls noise strength (timestep range) and clean patch ratio over training.
"""

import math


class CurriculumScheduler:
    """
    Manages the curriculum schedule for SubDiff pretraining.

    Over the course of training:
      - Noise timestep range [t_min, t_max] decays: strong noise → weak noise
      - Clean patch ratio decays: 25% → 5%

    Supports linear and cosine decay schedules.
    """

    def __init__(self, total_epochs,
                 t_min_start=800, t_min_end=100,
                 t_max_start=1000, t_max_end=600,
                 clean_ratio_start=0.25, clean_ratio_end=0.05,
                 warmup_epochs=10, schedule='cosine'):
        self.total_epochs = total_epochs
        self.t_min_start = t_min_start
        self.t_min_end = t_min_end
        self.t_max_start = t_max_start
        self.t_max_end = t_max_end
        self.clean_ratio_start = clean_ratio_start
        self.clean_ratio_end = clean_ratio_end
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule

    def _decay_factor(self, epoch):
        """Returns a factor in [0, 1] representing curriculum progress."""
        if epoch < self.warmup_epochs:
            return 0.0
        effective_epoch = epoch - self.warmup_epochs
        effective_total = self.total_epochs - self.warmup_epochs
        progress = min(effective_epoch / max(effective_total, 1), 1.0)

        if self.schedule == 'linear':
            return progress
        elif self.schedule == 'cosine':
            # Slow start, fast middle, slow end
            return 0.5 * (1 - math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def get_t_range(self, epoch):
        """Get the [t_min, t_max] range for noise timestep sampling."""
        factor = self._decay_factor(epoch)
        t_min = self.t_min_start + (self.t_min_end - self.t_min_start) * factor
        t_max = self.t_max_start + (self.t_max_end - self.t_max_start) * factor
        return int(t_min), int(t_max)

    def get_clean_ratio(self, epoch):
        """Get the clean patch ratio for this epoch."""
        factor = self._decay_factor(epoch)
        return self.clean_ratio_start + (self.clean_ratio_end - self.clean_ratio_start) * factor

    def get_state(self, epoch):
        """Get all curriculum parameters for the current epoch."""
        t_min, t_max = self.get_t_range(epoch)
        clean_ratio = self.get_clean_ratio(epoch)
        return {
            'epoch': epoch,
            't_min': t_min,
            't_max': t_max,
            'clean_ratio': clean_ratio,
            'decay_factor': self._decay_factor(epoch),
        }

    def __repr__(self):
        return (f"CurriculumScheduler(schedule={self.schedule}, "
                f"t_range=[{self.t_min_start},{self.t_max_start}]→"
                f"[{self.t_min_end},{self.t_max_end}], "
                f"clean_ratio={self.clean_ratio_start}→{self.clean_ratio_end})")
