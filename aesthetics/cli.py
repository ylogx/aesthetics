# -*- coding: utf-8 -*-
from aesthetics.fisher.cli import main

if __name__ == '__main__':
    import logging
    import sys

    logging.basicConfig(level=logging.DEBUG)
    sys.exit(main())
