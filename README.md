# telepath
Pytorch models for interpreting neuroimaging data.\
This project exists in orbit around one fundamental question: to what extent can the world model learnt during language model pretraining by transformers map to neural representations. To test this, I will start with a basic non-causal transformer encoder that generates embeddings which are concatenated onto the embeddings of the tokens generated by a frozen pre trained transformer that does the inference. Doing it this way gives a simple baseline to improve upon. It will also be interesting from a mechanistic interpreability standpoint to watch the generated embeddings move through token space as the encoder learns. The following questions will be of interest:

- How well does the model trained on the EEG data taken during the viewing of a set of objects generalise to the EEG data from an unseen set?
- How much do the EEG samples taken before or after the object was being viewed impact the accuracy of the model?
- the GPT will intially be frozen under the assumption that finetuning it will modify its representations of the out-of-distribution validation examples such that their logits are effectively never sampled...is that true?
- Are there more effective architectures? 

# TODO:
- [ ] Finish custom training code
    - [x] Fix logging
    - [x] Checkpointing
    - [ ] Custom metrics
        - [x] Naiive accuracy on object generation.
        - [x] Flexible accruracy (Capitalisations, synonyms; just decode and use string matching)
        - [ ] Semantic Similarity on object generations
- [ ] Architecture Ideas
    - [ ] COGVLM style neural decoder-only model: https://arxiv.org/pdf/2311.03079.pdf
        - [x] Expert Block
            - [x] networks
            - [x] routing
        - [x] Expert encoder
            - [x] Empty network
        - [x] Expert GPT
            - [x] transformer
            - [x] encoding
            - [x] routing
        - [x] new telepath
            - [x] weight freezing
        - [ ] tests
            - [x] expert GPT forward
            - [ ] routing
                - [ ] Assuming the attention matrices allow for it, do expert and core computations stat separate?
            - [x] generation
                - [x] create a test GPT which overrides forward with a deterministic computation and ensure the array slicing works correctly
            - [ ] weight freezing
                - [ ] are only expert params trainable
    - [ ] Pretrained encoder
    - [ ] Conv layers?
- [ ] Interp
    - [ ] Similarity of neural-expert-query to target-token-key vectors.
- [ ] Optimisatons
    - [x] Batch generate.
    - [ ] KV Cache.
    - [ ] Half precision training.
    - [ ] Multiple eval data loaders
    - [ ] Multi-worker dataloader.
- [ ] Training
    - [ ] DPO where negative preferences are selected stochastically from low semantic similarity objects
- [ ] Datasets
    - [ ] Data Augmentations
        - [ ] Duplicate rows with synoynms
        - [ ] Similate noise associated with eeg to create augmented rows.
        - [ ] Increase EEG window.
    - [ ] Multi-subject training set
    - [ ] Subject-held-out eval set
    - [ ] Subject+Object-held-out eval set
    
