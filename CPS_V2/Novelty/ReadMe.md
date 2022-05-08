# Adaptive Loss Weight for Cross Pseudo Supervision Loss

## Key Concept:
Main idea: The models are initially not accurate, and even after several epochs, the acuuracy fluctuates significantly. Hence, to use their outputs for cross training, especially when the number of unlabelled data is larger than that of labelled data (which means more iterations of training for unlabelled data), will hinder their learning and may reduce the models validation accuracy. 

Therefore, we propose an adaptive loss weight that multiplies the dice score from the supervised learning in that epoch to scale the unlabelled training loss using the accuracy of the model and hence, to reduce the gradients if the model is inaccurate (and vice-versa). 

Training Flow per epoch: 
  
  Supervised:
  train with labelled data
  use segmentation loss for model A and model B, and cps loss with fixed weight (e.g. 0.02)
  
  Unsupervised:
  train with unlabelled data
  use only cps loss between predictions of model A and model B
  cps loss has a fixed weight (e.g. 0.001), and multiply this with the average dice score from the supervised training in this epoch
  
