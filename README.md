# takeGo
This is partial and incomplete implementation of [Shodai AlphaGo (AlphaGo Fan paper)](https://vk.com/doc-44016343_437229031?dl=56ce06e325d42fbc72). Code was written in 2017 inspired by [Ray](https://github.com/kobanium/Ray) and [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo).

CGOS rating does not reach 2600.  
http://www.yss-aya.com/cgos/19x19/cross/take.html  
http://www.yss-aya.com/cgos/19x19/cross/mishima-0.1.html

## Playing Go
### Build container
```
docker build -t bbs -f ./docker/Dockerfile.tensorflow1.3.centos7 .
```
### Run GTP Server
```
docker run --rm --name bbs -p 5000:5000 bbs gtp -t 2 -lgrf2 --nogpu
```

### Playing with gogui
GoGUI is in this repository
```
java -jar tools/gogui-1.4.9/lib/gogui.jar
```
GoGUI command is set as follows
```
python /path/to/takeGo/bbc --host localhost --port 5000
```

## Development
### Build
```
docker exec -it bbs bash
cd ./gtp
python setup.py build_ext -i
```
### Run GTP server
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

## Training Networks
### Harvest patterns for rollout and tree policy
```
# Response Pattern (12-point diamond)
python bamboo/train/rollout/pattern_harvest_main.py -o /path/to/output/d12_rsp.csv -p d12_rsp -d /path/to/input/sgf/directory
# Non-Response Pattern (3x3)
python bamboo/train/rollout/pattern_harvest_main.py -o /path/to/output/x33.csv -p x33 -d /path/to/input/sgf/directory
# Non-Response Pattern (12-point diamond)
python bamboo/train/rollout/pattern_harvest_main.py -o /path/to/output/d12.csv -p d12 -d /path/to/input/sgf/directory
```
### Rollout Policy
```
# convert SGFs
python bamboo/train/rollout/sgf2hdf5_main.py -o /path/to/output/rollout_feature.h5 -d /path/to/input/sgf/directory -p rollout -mt ./params/rollout/mt_rands.txt -x33 /path/to/input/x33.csv -rd12 /path/to/input/d12_rsp.csv
# run training
python bamboo/train/rollout/supervised_rollout_trainer.py -p rollout /path/to/input/rollout_feature.h5 /path/to/weights/saved
```
### Tree Policy
```
# convert SGFs
python bamboo/train/rollout/sgf2hdf5_main.py -o /path/to/output/tree_feature.h5 -d /path/to/input/sgf/directory -p tree -mt ./params/rollout/mt_rands.txt -x33 /path/to/input/x33.csv -rd12 /path/to/input/d12_rsp.csv -d12 /path/to/input/d12.csv
# run training
python bamboo/train/rollout/supervised_rollout_trainer.py -p tree /path/to/input/tree_feature.h5 /path/to/weights/saved
```

## AlphaGo Papers
[Mastering the game of Go with deep neural networks and tree search](https://vk.com/doc-44016343_437229031?dl=56ce06e325d42fbc72)  
[Mastering the Game of Go without Human Knowledge](http://faculty.washington.edu/jwilker/559/2018/go.pdf)  
[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)  
[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
