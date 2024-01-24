#!/bin/bash
 
if [ "$1" = 'gtp' ]; then
    cd /games/gtp && python bbs \
        -pn ./params/policy/weights.hdf5 \
        -ro ./params/rollout/rollout.hdf5 \
        -tr ./params/rollout/tree.hdf5 \
        -mt ./params/rollout/mt_rands.txt \
        -x33 ./params/rollout/x33.csv \
        -rd12 ./params/rollout/d12_rsp.csv \
        -d12 ./params/rollout/d12.csv \
        --server \
        ${@:2} 
else
    exec "$@"
fi
