#!/bin/bash

cd /root/bambooStone
nohup python -u -m bamboo.train.distributed_supervised_policy_trainer --cluster_spec /cluster --job_name $(hostname | awk -F- '{print $1}') --task_index $(hostname | awk -F- '{print $2}') --logdir /tmp/logs --gpu_memory_fraction 0.2 > worker.log 2>&1 &

tail -f /dev/null
