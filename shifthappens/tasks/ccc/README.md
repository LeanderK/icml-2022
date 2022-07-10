# CCC: Continuously Changing Corruptions

## Task Description

CCC is a dataset used to benchmark models over time. The dataset is based on a generalization of common corruptions used in ImageNet-C, contains 462 million corrupted images in its development set, and pre-computed evaluation sets for evaluating continual adaptation algorithms over the course of one to ten million images to adapt to.

![image](https://user-images.githubusercontent.com/23415611/175703466-05631e4b-1cb7-4bdd-907a-85f737337b3a.png)


## Dataset Creation

Dataset creation is done by computing noise trajectories that maintain a constant baseline accuracy. 

## Expected Insights/Relevance

Using our dataset, we can rigorously test classification methods that continuously adapt to changing environments.
