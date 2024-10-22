# telepath
Adapting Whisper for reading neural signals. \
In short, my thought process is the following:
- Pretraining deep neural nets drastically increases their robustness to distributional drift, which is a problem that has plagued ML models trained on neuroimaging data.
- In absense of publically available in-domain pretrained models (or the resources to create one, yet), other studies have turned to pretrained image models and finetuned them on spectrograms.
- Surely transcribing audio is a much more closely matched task?

TODO:
[] Modify RPA to work with non-causal attention. \
    - make mod
    - make test
[] Implement T5 encoder. \
[] Implement neural encoder from first principles \
    - Keep linear proj \
    - Use RMS norm \
    - shift (but not scale) to t=0 before projection \
[] SLINP-Telepath \
    [] simese model definition \
        - if we just use THINGS then we can pre-compute the text embeddings
    [] SigLIP Loss fn \
    [] microbatch data swapping algorithm \
        - mbs are n_mb neu-text pairs \
        - texts are swapped between devices \
            - can just be indexes into the precomputed text emb matrix \
[] new val metrics \
    - accuracy @ n 

[] figure out what is going on with the generation tables?
