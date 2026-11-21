# Detecting Manipulated Images with Deep Learning

**George Washington University · Machine Learning II (DATS 6203)**

A collaborative academic machine-learning project investigating methods for distinguishing original photographs from digitally manipulated images using convolutional neural networks (CNNs), image preprocessing, transfer learning, and model ensembles.

**Contributors:** Mina Hanna and Mojahid Osman. This is a group project; work in this repository should not be attributed exclusively to one contributor.

## Problem and approach

Image manipulation can be difficult to identify from raw pixels alone. This project explores whether feature transformations and complementary deep-learning models can help classify images.

The documented experimental workflow includes:

1. **Prepare data** for original and manipulated image classes.
2. **Create transformed representations** using error level analysis (ELA), noise analysis, and gradient-based processing.
3. **Train models**, including custom CNNs and experiments with pretrained networks such as VGG19.
4. **Evaluate predictions** from saved models.
5. **Compare ensemble methods**, including hard voting and an MLP trained over model predictions.

This repository records experimental methods and coursework. It does not claim to be a production-ready image authentication service or a general detector of AI-generated images.

## Explore the repository

| Location | Contents |
| --- | --- |
| [`Code/`](Code/) | Preprocessing, CNN training, pretrained models, prediction, and ensemble scripts |
| [`Code/readme.md`](Code/readme.md) | Step-by-step data preparation and model workflow |
| [`Data/`](Data/) | Dataset acquisition and setup instructions |
| [`Group-Proposal/`](Group-Proposal/) | Project proposal |
| [`Final-Group-Project-Report/`](Final-Group-Project-Report/) | Group and individual reports |
| [`Final-Group-Project-Presentation/`](Final-Group-Project-Presentation/) | Presentation materials |

### Key Python modules

- `Data_Preprocessing.py` — dataset organization and image transforms
- `Modeling_CNN.py` — configurable custom CNN experiments
- `Modeling_Pretrained_CNN.py` — pretrained CNN experiments
- `Load_Predict_Measure.py` — model loading, prediction, and measurement
- `Hard_Voting_Committee.py` — ensemble majority voting
- `MLP_Ensembling.py` — neural-network ensemble combination

## Reproducing the experiments

Follow the instructions in [Data](Data/) and the [code workflow](Code/readme.md). Reproduction requires obtaining the specified datasets and configuring the Python environment; this academic repository is not packaged as a one-command application.

For model-specific results and experimental limitations, consult the [final reports](Final-Group-Project-Report/). No numerical performance claims are asserted in this overview without those reports.

## Research themes

Computer vision · Experimental machine learning · CNN architectures · Transfer learning · Model evaluation · Ensemble methods
