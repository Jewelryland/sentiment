# sentiment
Sentiment analysis on **Large Movie Review Dataset**

You can run the following commands to extract feature and train model
```shell
python feature.py
python train.py
```
This will perform a grid search on parameters of feature extracting and model training and is usually very slow.

With our pre-trained models in *models* directory, you can run `python demo.py` to see a demo classifies the positive and negative reviews stored in *pos_demo.txt* and *neg_demo.txt*
