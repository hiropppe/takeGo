# -*- coding:utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from bamboo.train.rollout import pattern_harvest

pattern_harvest.main(sys.argv[1:])
