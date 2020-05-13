# MADE Thousand Facial Landmarks

## Quick start

- Устанавливаем зависимости: <br>
```pip3 install -r requirements.txt```

- Качаем предобработанный ```cv2.resize(img, (180, 250), interpolation=cv2.INTER_LANCZOS4)``` датасет: <br>
```bash dataset.sh```

- Запускаем обучение: <br>
```catalyst-dl run --config=configs/train_1.yaml``` - resnet50, 14 epoch, batch_size 50, valid_size 0.2 <br>
```catalyst-dl run --config=configs/infer_1.yaml``` - создаем сабмит из лучшей модели. <br>
**PublicLB: 8.92621 <br>
PrivateLB: 8.68758 (13 место)**

- Дообучаем: <br>
```catalyst-dl run --config=configs/train_2.yaml``` - resnet50, 4 epoch, batch_size 50*3, valid_size 0.05 <br>
```catalyst-dl run --config=configs/infer_2.yaml``` - создаем сабмит из последней модели. <br>
**PublicLB: 9.27547 <br>
PrivateLB: 8.99571 (25 место)**

- Усредняем 2 сабмита: <br>
```python3 mean_subs.py```<br>
**PublicLB: 8.67260 <br>
PrivateLB: 8.42125 (7 место)**
![Лучший скор](images/best_score.png?raw=true "Лучший скор")

## Дополнительное описание
- Датасет был ресайзнут в 180x250, оригинальный размер сохранен в csv. На таком же разрешении училась модель.
- Аугментации из albumentations + свой HorizontalFlip(p=0.5)<br>
```
A.Compose([
    A.MotionBlur(p=0.2),
    A.OneOf([A.HueSaturationValue(p=0.5),
             A.RGBShift(p=0.5)], p=1),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensor()
])
```
- Использовал L2 loss как среднюю сумму дистанций между соответствующими точками
- Модель resnet50(pretrained=True) из torchvision
- [Catalyst](https://github.com/catalyst-team/catalyst)