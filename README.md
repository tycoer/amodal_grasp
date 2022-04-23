# RUN meshrcnn

server 24000 (dataset 也在24000)

## train
```buildoutcfg
cd amodal_grisp
python tools/train.py config/meshrcnn_r50_fpn_1x.py

or

bash tools/dist.sh config/meshrcnn_r50_fpn_1x.py 4

```

## test

```buildoutcfg
cd amodal_grisp
python test/test_evalution.py
```


