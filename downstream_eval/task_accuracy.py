from __future__ import annotations

try:
    from .model_accuracy_test import *  # noqa: F401,F403
    from .model_accuracy_test import main
except ImportError:
    from model_accuracy_test import *  # noqa: F401,F403
    from model_accuracy_test import main


if __name__ == "__main__":
    main()
