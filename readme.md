# Sparsity Verification for Attribution-Based Parameter Decomposition

A verification method for interpretability techniques that identify which neural network weights are responsible for specific classifications.

## Overview

This project validates Attribution-Based Parameter Decomposition (APD) results using an independent pairwise pruning approach. The core idea: if APD correctly identifies "cat-specific" weights, then selective pruning should find the same weights when isolating cat features across multiple binary tasks.

## Methodology

### Pairwise Pruning Pipeline
1. **Target Selection**: Focus on one class (e.g., "cat") in a multi-class classifier
2. **Binary Tasks**: Create multiple pairwise classifications:
   - Cat vs. Dog
   - Cat vs. Bird  
   - Cat vs. Horse
   - ... (~10 random pairings)
3. **Feature Intersection**: Combine results to isolate target-specific weights (eg just Cat related features)

The intuition being that Cat vs Dog will find the features that separate Cat vs Dog (eg cat ears, dog tail) while removing all irrelevant features (eg wings)


**Selective Pruning**: For each task:
   - Freeze base CNN weights
   - Apply learnable sparsity masks
   - Optimize for sparsity while maintaining binary performance

### Verification
Compare weight patterns from:
- **APD Method**: Weights identified when processing target images (eg the components that activate on cat pictures)
- **Pruning Method**: Weights retained after sparsity mask intersection