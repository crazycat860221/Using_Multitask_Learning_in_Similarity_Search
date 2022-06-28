# Using_Multitask_Learning_in_Similarity_Search
Chih-Yi Chiu, Ke-Long Zeng

# Abstract
In the task of nearest neighbor search, we present a method which is different from conventional approximate nearest neighbor search. The conventional search method calculates the distances between the query and cluster centroids, and then ranks clusters from near to far based on the distances. The data indexed in the topranked clusters are treated as the nearest neighbor candidates for the query. The probability-based ranking can alleviate the influence of the loss of quantization between the data and cluster centroids on distance-based ranking. Based on probability-based ranking, we use triplet loss to train the query embeddings to improve the probability prediction. Furthermore, our model predicts the amount of candidates needed to find the nearest neighbor for each query to reduce the computation cost. We apply multi-task learning to train the model. We experimented on two large-scale nearest neighbor search datasets and evaluated our model based on top-k recall and the number of candidates. Our experimental results demonstrated that the proposed method could improve the performance of probability-based ranking effectively.

Index Terms — approximate nearest neighbor search; early termination; triplet loss

# Source Code
Our source codes and data for SIFT1B RVQ 4096x4096 is available at:
https://drive.google.com/drive/folders/1X42cO1MItkln0J2Jc98sARrLGxh55mVx?usp=sharing

IVF file is available to download from:
https://drive.google.com/drive/folders/15Xuk9jEEcZdD0x0ECz2PuCoZuw0RHoQf?usp=sharing

These codes are in the coarse search process. We simulate how to retrieve the candidate second-level clusters for SIFT1B RVQ before performing the ADC. Then, we calculate the top-1 recall along with the candidate second-level clusters.
  
