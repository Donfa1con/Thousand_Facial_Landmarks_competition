# MADE Thousand Facial Landmarks

## Quick start

- Устанавливаем зависимости: <br>
```pip3 install -r requirements.txt```

- Качаем предобработанный ```cv2.resize(img, (180, 250), interpolation=cv2.INTER_LANCZOS4)``` датасет: <br>
```bash dataset.sh```

- Запускаем обучение: <br>
```catalyst-dl run --config=configs/train_1.yaml``` - resnet50, 14 epoch, batch_size 50, valid_size 0.2 <br>
```catalyst-dl run --config=configs/infer_1.yaml``` - создаем сабмит из лучшей модели. <br>
**PublicLB: 8.68758 <br>
PrivateLB: 8.92621 (21 место)**

- Дообучаем: <br>
```catalyst-dl run --config=configs/train_2.yaml``` - resnet50, 4 epoch, batch_size 50*3, valid_size 0.05 <br>
```catalyst-dl run --config=configs/infer_2.yaml``` - создаем сабмит из последней модели. <br>
**PublicLB: 8.99571 <br>
PrivateLB: 9.27547 (37 место)**

- Усредняем 2 сабмита: <br>
```python3 mean_subs.py```<br>
**PublicLB: 8.42125<br>
PrivateLB: 8.67260 (7 место)**
