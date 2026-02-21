import numpy as np
from verl.trainer.ppo.custom_sampler import CurriculumSampler


class SkyRLCurriculumSampler(CurriculumSampler):
    """
    Wraps verl's CurriculumSampler with a stateful __iter__ so that each
    sample is yielded exactly once per epoch.  The parent class is stateless:
    _select_closest_to_target always returns the same top-K indices, causing
    every batch in an epoch to be identical.  This subclass fixes that by
    maintaining a pool of unused indices and drawing from it each step.
    """

    def __iter__(self):
        # Build a pool of all indices sorted by distance to current target.
        diffs = np.abs(self.difficulties - self.target_difficulty)
        pool = list(np.argsort(diffs))  # closest-first ordering

        while len(pool) >= self.batch_size:
            batch = pool[: self.batch_size]
            pool = pool[self.batch_size :]
            yield [int(i) for i in batch]

    def __len__(self):
        return len(self.data_source) // self.batch_size
