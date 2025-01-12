# takeGo
This is partial and incomplete implementation of [Shodai AlphaGo (AlphaGo Fan paper)](https://vk.com/doc-44016343_437229031?dl=56ce06e325d42fbc72). Code was written in 2017 inspired by [Ray](https://github.com/kobanium/Ray) and [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo).

CGOS rating does not reach 2600.  
http://www.yss-aya.com/cgos/19x19/cross/take.html  
http://www.yss-aya.com/cgos/19x19/cross/mishima-0.1.html 

***(Re)development branch***

## Development
### Build
```
python3 setup.py build_ext -i
```
## Training Networks
### Supervised Learning Policy
```
# convert SGFs
python3 -m bamboo.scripts.policy_feature -o /path/to/feature_planes.h5 -d /path/to/sgf/directory
# run training
python3 -m bamboo.scripts.keras_supervised_policy_trainer train /path/to/weights/saved /path/to/feature_planes.h5
```
### Patterns for rollout and tree policy
```
# Response Pattern (12-point diamond)
python3 -m bamboo.scripts.rollout_pattern -o /path/to/d12_rsp.csv -p d12_rsp -d /path/to/sgf/directory
# Non-Response Pattern (3x3)
python3 -m bamboo.scripts.rollout_pattern -o /path/to/x33.csv -p x33 -d /path/to/sgf/directory
# Non-Response Pattern (12-point diamond)
python3 -m bamboo.scripts.rollout_pattern -o /path/to/d12.csv -p d12 -d /path/to/sgf/directory
```
### Rollout Policy
```
# convert SGFs
python3 -m bamboo.scripts.rollout_feature -o /path/to/rollout/feature.h5 -d /path/to/sgf/directory -p rollout
# run training
python3 -m bamboo.scripts.supervised_rollout_trainer -p rollout /path/to/rollout/feature.h5 /path/to/weights/saved
```
### Tree Policy
```
# convert SGFs
python3 -m bamboo.scripts.rollout_feature -o /path/to/tree/feature.h5 -d /path/to/sgf/directory -p tree
# run training
python3 -m bamboo.scripts.supervised_rollout_trainer -p tree /path/to/tree/feature.h5 /path/to/weights/saved
```

## AlphaGo Papers
[Mastering the game of Go with deep neural networks and tree search](https://vk.com/doc-44016343_437229031?dl=56ce06e325d42fbc72)  
[Mastering the Game of Go without Human Knowledge](http://faculty.washington.edu/jwilker/559/2018/go.pdf)  
[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)  
[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
