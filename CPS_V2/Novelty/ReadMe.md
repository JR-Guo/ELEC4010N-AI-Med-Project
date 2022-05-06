# Adaptive Loss Weight for Cross Pseudo Supervision Loss

## Key Concept:

Logic Flow

for epoch in max_epoch:
  
  Supervised:
  train with labelled data
  use segmentation loss for model A and model B, and cps loss with fixed weight (e.g. 0.02)
  
  Unsupervised:
  train with unlabelled data
  use only cps loss between model A and model B
  cps loss has a fixed weight (e.g. 0.001), and multiply this with the average dice score from the supervised training in this epoch
  
Main idea: if the model is less accurate, then cross train the models less (because they are not accurate enough themselves)
