#Receipt-recognition

Extract key information of a receipt image

##Train text detection model
```
python3 train_detector.py -d <dir of the dataset>
```
Arguement:
```
usage: train_detector.py [-h] [--dataset [{coco_text,receipt,synthtext}]] -d
                         DIR [-b BATCH_SIZE] [-i IMAGE_SIZE] [-s]

Train detection model

optional arguments:
  -h, --help            show this help message and exit
  --dataset [{coco_text,receipt,synthtext}]
                        Enter the dataset
  -d DIR, --dir DIR     Directory of dataset
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -i IMAGE_SIZE, --image_size IMAGE_SIZE
                        Reshape size of the image
  -s                    Shut down vm after training stop
```

##Train receipt extraction model
```
python3 train_grid_classifier.py -d <dir of the dataset>
```

Arguement:
```
usage: train_grid_classifier.py [-h] [-e] [-b BATCH_SIZE] -d DIR [-s]

Train detection model

optional arguments:
  -h, --help            show this help message and exit
  -e, --emb             Train character embedding layer
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -d DIR, --dir DIR     Directory of dataset
  -s                    Shut down vm after training stop
```
