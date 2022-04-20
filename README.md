# <div align="center">Semantic Segmentation</div>
</div>

1. Build Docker Image

```bash
$ docker build -t segmgentation -f Dockerfile .
```

2. Run the inference model

```bash
$ python tools/infer.py --cfg configs/custom.yaml
```

3. Run the script to check the received masks

```bash
$ python test_segformer.py
```
