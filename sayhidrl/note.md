Example structure of datasets required:
```
your_dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── annotations/
    └── train.json
```


python train_custom_dataset.py --config-file ../config.yaml --cpu --eval-only --output-dir FinalLocalEval/ MODEL.WEIGHTS model_final.pth 