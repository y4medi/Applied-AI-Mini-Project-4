# Fashion Classifier - StyleSort Project

**Team Members: Yansong Jia, Sepehr Mansouri**

**Mini Project 4 for COMP 9130 Applied AI**

## 1. Business Problem
**StyleSort Inc.**, an online fashion retailer, faces a critical operational challenge: a 32% return rate, largely driven by product miscategorization. Manual sorting is error-prone and costly.
*   **Operational Cost**: Returns cost $8-12 per item in shipping and processing.
*   **Customer Experience**: Misclassifying a *Coat* as a *Pullover* leads to unmet expectations regarding warmth, while confusing a *Shirt* for a *T-shirt* frustrates customers seeking formal attire.
The goal is to automate classification with >85% accuracy while minimizing these high-impact errors.

## 2. Methodology
We implemented a Convolutional Neural Network (CNN) using **PyTorch** on the Fashion-MNIST dataset (28x28 grayscale images).

### Architectures Evaluated
1.  **Baseline**: Single hidden layer (512 units), ReLU, 50% Dropout.
2.  **Deep Network**: Three hidden layers (512 -> 256 -> 128) to learn complex hierarchal features.
3.  **Batch Normalization (Best)**: Deep network with Batch Norm and Leaky ReLU. This configuration stabilized training, allowing for a lower dropout rate (30%) and higher learning capacity.

### Training Procedure
*   **Optimizer**: Adam (Adaptive Moment Estimation) for efficient convergence.
*   **Loss Function**: CrossEntropyLoss.
*   **Protocol**: 10 epochs, Batch size 64, with automated checkpointing of the best model based on validation accuracy.

## 3. Results Summary
The **Batch Norm + Leaky ReLU** model achieved a final test accuracy of **92.86%**, significantly exceeding the 85% requirement.

### Key Metrics
*   **Confusion Matrix Analysis**:
    *   **Major Error**: *Shirt* vs. *T-shirt/Top* (164 errors). These categories distinguish poorly at low resolution.
    *   **Secondary Error**: *Coat* vs. *Pullover* (89 errors). Visual similarity leads to confusion.
    *   **Strength**: Footwear (Sneakers, Sandals) and Bags achieved >98% accuracy.
*   **Cost-Weighted Analysis**:
    *   Assigning higher penalties to costly errors (e.g., Coat/Pullover) revealed that the top two error pairs account for **65% of the total error cost**. Optimization here yields the highest ROI.
*   **Confidence Threshold**:
    *   Optimal Threshold: **0.95**.
    *   Impact: **83.6%** of items are auto-classified with **97.97% accuracy**. Only **16.4%** flagged for manual review.

## 4. Business Recommendations
1.  **Implement Hybrid Workflow**: Deploy the model with a 0.95 confidence threshold. This automates the vast majority of sorting while routing ambiguous items (mostly Shirts/Coats) to human experts, reducing the error rate from ~7% to ~2%.
2.  **Targeted Data Improvement**: The visual distinction between Shirts and T-shirts is the primary bottleneck. We recommend updated photography guidelines to emphasize collars and button details.
3.  **Strategic Metadata**: For Coats and Pullovers, require additional metadata (e.g., "Closure Type", "Material Weight") during upload to aid classification.
4.  **Financial Impact**: Reducing misclassifications by ~4,300 items monthly saves estimated **$774,000 annually** in operational costs.

## 5. Setup and Running Instructions
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Experiments**:
    Open `notebooks/fashion_classifier.ipynb` in Jupyter/VS Code and Run All Cells. This will reproduce the training, evaluation metrics, and visualization plots.

## Repository Structure
```
fashion-classifier/
├── notebooks/
│   └── fashion_classifier.ipynb  # Main analysis
├── src/
│   ├── model.py                  # CNN architectures
│   ├── train.py                  # Training pipeline
│   └── utils.py                  # Data loading & plotting
├── results/                      # Generated plots
├── requirements.txt
└── README.md
```

## 6. Team Contributions
*   **Yansong Jia**: Project Infrastructure, Model Engineering (`model.py`), Training Pipeline (`train.py`), Analytical Backend (`utils.py`), and README.
*   **Sepehr Mansouri**: Model Benchmarking, Visualizations, Business Analysis (`notebooks/fashion_classifier.ipynb`), and Strategic Reporting.
