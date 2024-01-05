## clustering algorithms that assign to multiple clusters

## reduce dimensionality before clustering
SVD, curse of dimensionality 

## number of clusters:
- K-Means
    - Elbow method
- Siloette method

## topic extraction
- LDA

## assign to multiple clusters
- above 33%

## presentation
- show how articles are assigned to primary and secondary clusters

primary and secondary clusters
The GaussianMixture is very certain wrt it's predictions as to the category within which the article belongs.

articles with predictions between 0.005 - 0.995 are given secondary, tertiary etc. clusters with which they are also labelled.

The data which have been assigned secondary clusters should have a different colour when plotted so that we can hopefully see on the scatter plot, after PCA/SVD that they are there where the clusters meet/overlap.

- **I also need to keep track of the probability of the 2nd cluster**
- all the different methods for determining the number of clusters,
    - silhouette
    - elbow
    - log of GMM

- present the results
    - clusters
    - articles with secondary clusters

Compare how they were grouped/categorised and how they are now, similar, overlapping, what are the major changes, combinations?

preprocess, split into groups of 10000 lines and then concat afterwards