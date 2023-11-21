# telepath
Pytorch models for interpreting neuroimaging data.\
This project exists in orbit around one fundamental question: to what extent can the world model learnt during language model pretraining by transformers map to neural representations. To test this, I will start with a basic non-causal transformer encoder that generates embeddings which are concatenated onto the embeddings of the tokens generated by a frozen pre trained transformer that does the inference. Doing it this way gives a simple baseline to improve upon. It will also be interesting from a mechanistic interpreability standpoint to watch the generated embeddings move through token space as the encoder learns. The following questions will be of interest:

- How well does the model trained on the EEG data taken during the viewing of a set of objects generalise to the EEG data from an unseen set?
- How much do the EEG samples taken before or after the object was being viewed impact the accuracy of the model?
- the GPT will intially be frozen under the assumption that finetuning it will modify its representations of the out-of-distribution validation examples such that their logits are effectively never sampled...is that true?
- Are there more effective encoder architectures? (e.g., Meta's wave2vec2 with a pre-transformer convolution)?
    - COGVLM but with a neural expert instead of a visual expert: https://arxiv.org/pdf/2311.03079.pdf


# TODO:
- [ ] Finish custom training code
    - [x] Fix logging
    - [ ] Checkpointing
    - [ ] Custom metrics (e.g., Accuracy on object generation)
- [ ] Architecture Ideas
    - [ ] COGVLM style neural decoder-only model
    - [ ] Conv layers?
- [ ] Interp
    - [ ] On base E-D Telepath
        - [ ] Nearest-Neighbour token space tracking of encoder outputs
    - [ ] On COGVLM
        - [ ] Similarity of neural-query to target-token-key vectors.
- [ ] Optimisatons
    - [ ] Multiple eval data loaders
    - [ ] Multi-worker dataloader.
- [ ] Datasets
    - [ ] Multi-subject training set
    - [ ] Subject-held-out eval set
    - [ ] Subject+Object-held-out eval set
    
