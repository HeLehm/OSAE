# What to evaluate:
## SPARSE AUTOENCODERS FIND HIGHLY INTERPRETABLE FEATURES IN LANGUAGE MODELS
### INTERPRETABILITY AT SCALE
ask a LLm what this feature means (Bills et al. (2023))

### Activation Patching
- edit the model’s internal activations along the directions indicated by our dictionary features and measure the changes to the model ’s outputs.
- Indirect Object Identification (IOI)

### investigate individual features (hard to compare)
- (1) Input: We identify which tokens activate the dictionary feature and in which contexts, (2) Output: We determine how ablating the feature changes the output logits of the model, and (3) Intermediate features: We identify the dictionary features in previous layers that cause the analysed feature to activate.
- hard to compare when for example SAE has this feature,and the cricuit doesn't have it, but the circuit has another feature that SAE doesn't have.

### Fraction of variance explained vs avg number of activated features 
- have not found a good explination here yet

## PURE
### qulaitativle show samplesa in clusters
- find poylsemantic neurons
- cluster based on PURE features and calculate intra and inter cluster similarity


## SAE initialization
- ref: https://www.lesswrong.com/posts/YJpMgi7HJuHwXTkjk/taking-features-out-of-superposition-with-sparse
- just compare the cosine dista of in and ouput
- idea in our case: initialized based on neurons that have a lot of differnt circuits activating them