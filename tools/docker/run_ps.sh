#!/bin/bash

cd /root/bambooStone
nohup python -u -m bamboo.train.distributed_supervised_policy_trainer --cluster_spec /cluster --job_name $(hostname | awk -F- '{print $1}') --task_index $(hostname | awk -F- '{print $2}') --gpu_memory_fraction 0.1 > ps.log 2>&1 &

tail -f /dev/null
