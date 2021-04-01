import os
import resource

from wasabi import Printer

# The Wasabi Printer object used throught the repo
# See: https://github.com/ines/wasabi for details.
msg = Printer(line_max=100)

# Set a stricter threshold for which Tensorflow logs to surface at the command line.
# See: https://stackoverflow.com/a/42121886/6578628 for details on this solution.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa


__version__ = "0.1.0"
