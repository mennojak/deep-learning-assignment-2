# Deep Learning Assignment 2: MEG Classification

This project implements various deep learning and machine learning approaches for classifying MEG (magnetoencephalography) data. The assignment involves building models for both intra-subject and cross-subject classification tasks using different feature representations. The implementation explores both classical machine learning algorithms and modern deep learning architectures (CNN, EEGNet, Graph Attention Networks) to compare their effectiveness on the MEG classification problem.

## Project Structure
- **Notebooks**: Main notebooks for preprocessing (`load_and_prep_data.ipynb`), training (`cross_classification.ipynb`, `intra_classification.ipynb`) and analysis (`comparison.ipynb`).
- **Data**: Preprocessed MEG data in `.npy` format and raw data in HDF5 format, both omitted in git because of their size.
- **Experiments**: Additional exploratory notebooks testing various architectures and techniques.
- **Bandpower data**: Pre-computed bandpower features for faster model training and evaluation.
