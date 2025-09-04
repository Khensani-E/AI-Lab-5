# AI-Lab-5
## Lab 5: Dimensionality Reduction
This lab explores dimensionality reduction techniques including Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Kernel PCA (KPCA).

### Objectives
• Understand how PCA maximizes variance for unsupervised dimensionality reduction.

• Apply LDA to maximize class separability using label information.

• Explore KPCA for nonlinear feature mappings using the RBF kernel.

• Compare performance of PCA, LDA, and KPCA on real (Wine dataset) and synthetic (half-moons, circles) datasets.

• Visualize transformed datasets and evaluate classification performance (e.g., Logistic Regression).
### Prerequisites
• A Google account

• Google Colab environment

• Required libraries are already available in Colab:
-numpy
-pandas
-matplotlib
-scikit-learn
-scipy

If you need to install or upgrade a package in Colab, use:

!pip install numpy pandas matplotlib scikit-learn scipy
### Getting Started (Google Colab)

1.Open Colab: https://colab.research.google.com/

2.Upload the provided notebook file (dimensionality_reduction_lab.ipynb) or create a new notebook.

3.If using the Wine dataset, download directly in Colab:

!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data -O wine.data


4.Run the cells step by step.

### Lab Outline
#### Part 1: Principal Component Analysis (PCA)

- Load and preprocess the Wine dataset.

- Implement PCA from scratch (covariance, eigen decomposition).

- Plot explained variance ratios.

- Compare with scikit-learn PCA.

- Train Logistic Regression on PCA-transformed data and visualize decision regions.

~ Expected Outcome: PCA reduces data dimensionality, capturing variance but not necessarily class separability.

#### Part 2: Linear Discriminant Analysis (LDA)

- Compute within-class and between-class scatter matrices.

- Solve the generalized eigenvalue problem manually.

- Compare with scikit-learn LDA.

- Train Logistic Regression and visualize decision regions.

~ Expected Outcome: LDA uses label information to maximize separability, leading to higher classification accuracy.

#### Part 3: Kernel Principal Component Analysis (KPCA)

- Implement RBF Kernel PCA from scratch.

- Apply KPCA to half-moon dataset and concentric circles dataset.

- Compare with scikit-learn KernelPCA.

~ Expected Outcome: KPCA makes nonlinear data linearly separable in feature space.

#### Analysis Questions

1.Explained Variance (PCA):

- The first few principal components (PCs) contribute the most to the dataset’s variance. You can see steep bars for PC1, PC2, and PC3, which then taper off — classic “elbow” behavior.
- This curve steadily climbs toward 1.0, meaning that as you add more components, you retain more of the original data’s variance. The curve flattens after a few components, signaling diminishing returns.
- From the graph:

The cumulative line crosses the 0.95 threshold around PC9 or PC10.

That means 9–10 components are sufficient to retain 95% of the dataset’s variance.

2.PCA vs. LDA:

- PCA Projection:

Projects data onto directions of maximum variance.

Class clusters may overlap if variance doesn’t align with class boundaries.

Logistic regression decision boundaries are broader and less crisp.

- LDA Projection:

Projects data onto axes that maximize inter-class separation.

Clear, well-separated clusters with tight decision boundaries.

Logistic regression performs better due to reduced intra-class variance and enhanced separability.

-   Why LDA Typically Outperforms PCA for Classification
Class Awareness: LDA uses label information to find the most discriminative features — PCA doesn’t.

Dimensionality Reduction with Purpose: LDA reduces dimensions while preserving class structure, which is ideal for classifiers.

Improved Decision Boundaries: As your LDA-transformed plot shows, logistic regression draws cleaner, more accurate boundaries.

Noise Filtering: LDA suppresses variance that doesn’t help distinguish classes — PCA retains all variance, even irrelevant.

3.KPCA Gamma Parameter:

- γ = 0.01 → Underfitting
The transformed data is only mildly curved, and the two classes (red triangles and blue circles) are not well separated.

The kernel is too smooth — it fails to capture the nonlinear structure of the half-moon shapes.

Linear separability is weak, and classifiers like logistic regression would struggle to draw clean boundaries.

This is underfitting: the transformation is too coarse to reveal meaningful structure.

- γ = 100 → Overfitting
The transformed data is highly warped, with tight clusters and exaggerated separation.

The kernel is too sharp — it overreacts to small differences between points.

While the classes appear more separated, the transformation may be too sensitive to noise, leading to poor generalization.

This is overfitting: the kernel captures fine-grained details that don’t help with robust classification.

4.Classifier Performance:

- Compare Logistic Regression accuracy on:

  -- Original Data: High accuracy, but slower due to full feature space.

-- PCA: Slight drop in accuracy, faster training, no label awareness.

-- LDA: Highest accuracy and fastest training — optimized for class separation.

5.Limitations:

- It fails when the data has nonlinear patterns. A classic example from the lab:

Example: Half-Moon Dataset
Two interleaved half-circle clusters.

PCA projects them onto a linear subspace, but the classes remain entangled.

Logistic regression on PCA-transformed data struggles to separate them.

This is because PCA can’t “unfold” the curved geometry — it treats the moons as overlapping blobs.

- Kernel PCA (KPCA) extends PCA by applying the kernel trick:

It maps data into a higher-dimensional feature space using a nonlinear kernel (e.g., RBF).

Then it performs PCA in that space, capturing nonlinear relationships.

Example: KPCA on Half-Moon Data
With γ = 15, KPCA transforms the moons into a space where they become linearly separable.

Logistic regression now draws clean decision boundaries.

The transformed scatter plot shows distinct clusters — a huge improvement over standard PCA.

#### Summary

- PCA reduces variance dimensions but doesn’t optimize for class labels.

- LDA performs better for supervised classification tasks.

- KPCA handles nonlinear data structures effectively.
