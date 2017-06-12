# -*- coding:utf-8 -*-

import sys

from bamboo.rollout import pattern_harvest
print ' '.join(sys.argv)
pattern_harvest.main(sys.argv[1:])
