# ELEC4010N Final Group Project

## 3D Semi-Supervised Segmentation on Left-Atrium MRIs with Cross Pseudo Supervision and Mean Teacher model with Uncertainty Map

#### Group Members: Guo Jiarong, Gupta Pranav

### Key Objectives:

* Train 3D Supervised Segmentation model with 20%/100% Labelled Data
* Use two 3D Semi-Supervised Segmentation learning methods and compare results
   - Cross Pseudo Supervision
   - Uncertainty Aware Mean Teacher Model
* Novelties
   - Lovasz-Softmax Loss for Boundary-Smoothening
   - Adaptive CPS Loss Weight 

The codes relevant to Cross Pseudo Supervision are inside the folder CPS_V2.
The folder Novelties inside CPS_V2 contains the code for the Novelty No.2: Adaptive CPS Loss Weight

The codes relevant to Mean Teach model are inside the folder MT. 
(Novelty codes)

Codes outside the folders are either relevant to Q1 - Fully Supervised Learning, or debugging code that can be ignored. 
