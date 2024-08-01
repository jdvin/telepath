# telepath
Adapting Whisper for reading neural signals. \
In short, my thought process is the following:
- Pretraining deep neural nets drastically increases their robustness to distributional drift, which is a problem that has plagued ML models trained on neuroimaging data.
- In absense of publically available in-domain pretrained models (or the resources to create one, yet), other studies have turned to pretrained image models and finetuned them on spectrograms.
- Surely transcribing audio is a much more closely matched task?

TODO:
[] val batch size
[] Separate conv for every electrode
    [] can then have avg conv weight norm for each electrode as a metric
[] multipe epochs
[] figure out what is going on with the generation tables?
