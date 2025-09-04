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

- How many components are needed to explain ~95% of variance?

2.PCA vs. LDA:

- Which gives better separation for the Wine dataset? Why?

3.KPCA Gamma Parameter:

- What happens if γ is too small or too large?

4.Classifier Performance:

- Compare Logistic Regression accuracy on:

  -- Original dataset
  -- PCA-transformed dataset
  --LDA-transformed dataset

5.Limitations:

- When does PCA fail?

- How does KPCA fix this?

#### Summary

- PCA reduces variance dimensions but doesn’t optimize for class labels.

- LDA performs better for supervised classification tasks.

- KPCA handles nonlinear data structures effectively.
