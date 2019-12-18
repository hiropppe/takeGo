# takeGo
This is partial and incomplete implementation of [Shodai AlphaGo (AlphaGo Fan paper)](https://vk.com/doc-44016343_437229031?dl=56ce06e325d42fbc72). Code was written in 2017 inspired by [Ray](https://github.com/kobanium/Ray) and [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo) but it's not very good code. I rewrite it when I have time for my old age:-)

CGOS rating does not reach 2600.  
http://www.yss-aya.com/cgos/19x19/cross/take.html  
http://www.yss-aya.com/cgos/19x19/cross/mishima-0.1.html


The following is a note for myself.

## Usage
### Containers
```
git clone https://github.com/hiropppe/takeGo.git
cd takeGo/tools/docker
# CPU machine
docker build -t takeGo -f Dockerfile.tensorflow1.3.centos7 .
docker run -td --name takeGo --net host takeGo /bin/bash
# GPU machine
docker build -t takeGo -f Dockerfile.tensorflow1.3.cuda8.0.cudnn6.ubuntu16.04 .
docker run -td --runtime nvidia --name takeGo --net host takeGo /bin/bash
```
### Build 
```
python setup.py build_ext -i
```
### Run GTP server
Required to image_data_format in ~/.keras/keras.json. NHWC(channels_last) for CPU and NCHW(channels_first) for GPU. (not dockernized)
```
python bbs \
  -pn ./params/policy/weights.hdf5 \
  -ro ./params/rollout/rollout.hdf5 \
  -tr ./params/rollout/tree.hdf5 \
  -mt ./params/rollout/mt_rands.txt \
  -x33 ./params/rollout/x33.csv \
  -rd12 ./params/rollout/d12_rsp.csv \
  -d12 ./params/rollout/d12.csv \
  -t 10 \
  -lgrf2 \
  --nogpu \
  --server

# --nogpu (CPU only)
```
### GoGUI command
```
python /path/to/takeGo/bbc --host {docker_host} --port 5000
```
## Training
### SL Policy
TODO
### Rollout and Tree policy
TODO
