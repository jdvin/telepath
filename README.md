# telepath
Attempting small scale EEG model pretraining with contrastive learning.

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
- [x] SLINP-Telepath
    - [x] simese model definition
        - if we just use THINGS then we can pre-compute the text embeddings
    - [x] SigLIP Loss fn
    - [x] microbatch data swapping algorithm
        - mbs are n_mb neu-text pairs
        - texts are swapped between devices
            - can just be indexes into the precomputed text emb matrix
- [x] Contrastive data handling
    - we have multiple presentations per object, but we cannot have two presentations of the same objects 
    be in the same batch.
        - or could we just have multiple +1 items in the loss fn?
- [x] new val metrics
    - accuracy @ n 
- [x] Make Training Work Again!
- [x] Swap to sentence t5
- [x] subject embeddings
- [ ] More comprehensive test suite
- [ ] Training state checkpointing and resumption
- [ ] Add resting state data
- [ ] Swap AdamW to SOAP
- [ ] add THINGS-EEG data (50ms)
    - add shift for forward so that epoch always begins at constant point
- [ ] Clean up some inefficiencies
    - Actual batch passing, rather than just an all reduce
- [ ] figure out what is going on with the generation tables?
