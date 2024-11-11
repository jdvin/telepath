# telepath
Adapting Whisper for reading neural signals. \
In short, my thought process is the following:
- Pretraining deep neural nets drastically increases their robustness to distributional drift, which is a problem that has plagued ML models trained on neuroimaging data.
- In absense of publically available in-domain pretrained models (or the resources to create one, yet), other studies have turned to pretrained image models and finetuned them on spectrograms.
- Surely transcribing audio is a much more closely matched task?

## TODO:
- [x] Modify RPA to work with non-causal attention.
- [x] Implement T5 encoder.
    - [x] Make test.
        - Auto model divergence analyser
            - receive maps from module names between models, and input/output to tensor functions, and install hooks to save the tensors and then load and compare them for each module
- [x] Implement neural encoder from first principles.
        - Keep linear proj.
        - Use RMS norm.
        - shift (but not scale) to t=0 before projection.
- [ ] SLINP-Telepath
    - [x] simese model definition
        - if we just use THINGS then we can pre-compute the text embeddings
    - [x] SigLIP Loss fn
    - [ ] microbatch data swapping algorithm
        - mbs are n_mb neu-text pairs
        - texts are swapped between devices
            - can just be indexes into the precomputed text emb matrix
- [ ] new val metrics
    - accuracy @ n 
- [ ] add THINGS-EEG data (50ms)
    - add shift for forward so that epoch always begins at constant point

- [ ] figure out what is going on with the generation tables?
