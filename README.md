# DRUIC 
Implementation of "Fine-Grained Multi-Aspect Matching between Disentangled Representations of User Interests and Content for News Recommendation" (DASFAA 2024).

## Dataset
The origin dataset can be downloaded from https://msnews.github.io/index.html

For preprocessing of dataset utils, please refer to https://github.com/recommenders-team/recommenders/blob/main/examples/01_prepare_data/mind_utils.ipynb.

## Baselines

In this repository, we exclusively focus on news recommendation baselines and do not include traditional recommendation methods.

For some of the baselines, such as **DKN, NPA, LSTUR, and NRMS**, we utilized code from the https://github.com/recommenders-team/recommenders. 

For some other baselines which the authors had published the code, we utilized publicly available code. The following are the code links for these baselines:

**FUM** : https://github.com/taoqi98/FUM  \
**CAUM** : https://github.com/taoqi98/CAUM \
**FIM** : https://github.com/taoqi98/FIM \
**MINS** : https://github.com/whonor/MINS \
**MIECL** : https://github.com/wangsc2113/MIECL 


For the baseline without publicly available code, we reproduced it based on its paper, and you can find the reproduced code in the **Baselines** folder, which including **MCCM** and **MINER**. For fair comparison, we replace the BERT news encoder within MINER with pretrained Glove embedding vectors and self-attention networks.

## Additional Experiments
Due to page limitations, we removed some experiments from the paper, which we show below:

### Hyperparameter Experiment

From left to right: experiment on loss function weight hyperparameters and experiment on the number of aspects (K) hyperparameters.

<img src="./figures/lamada.png" alt="Editor" width="250">        <img src="./figures/K.png" alt="Editor" width="250">

### Visualization of Different Disentanglement Loss

From left to right: No Disentanglement Loss, Orthogonal Disentanglement Loss, and DRUIC: 

<img src="./figures/Without dis.png" alt="Editor" width="250"><img src="./figures/Orthogonal.png" alt="Editor" width="250"><img src="./figures/ours.png" alt="Editor" width="250">

### Case Study

Below is a visualization showing the matching scores between a user's interests and various aspects of news articles. The darker the color, the greater the user's interest in that particular aspect of the news.

<img src="./figures/case_study.png" alt="Editor" width="700">
