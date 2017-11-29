import numpy as np
from nolearn.lasagne import BatchIterator


# TODO add more rotations degree and contranst funct
class FlipBatchIterator(BatchIterator):

    def transform(self, xb, yb):
        xb, yb = super(FlipBatchIterator, self).transform(xb, yb)
        # TODO use rotate from scipy
        bs = xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)  # /2 choose all
        xb[indices] = xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

        return xb, yb
