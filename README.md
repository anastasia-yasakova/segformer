# <div align="center">Semantic Segmentation</div>
</div>

1. Clone the repository

```bash
$ git clone https://github.com/anastasia-yasakova/segformer
$ cd segformer
```

2. Build Docker Image

```bash
$ docker build -t segmentation -f Dockerfile .
```

3. Run the inference model

```bash
$ python tools/infer.py --cfg configs/custom.yaml
```

4. Run the script to check the received masks

```bash
$ python test_segformer.py
```
