## Overview
This repository provides the codes to reproduce the results from *Jianping Ye, Michel Wedel, Pieters, AdGazer: Improving Contextual Advertising with Theory-Informed Machine Learning*. We propose an *Alternative Disclosure Plan* for a delay of the release of our main dataset of the ad-context pairs. However, to maximize the reproducibility of the paper's results in this situation, we provide:

- All preprocessed data used in the paper for model training, i.e. all features extracted from ad and context images in our main dataset with our algorithms;
- All codes for the algorithms and models used in the paper;
- All trained models (Sentence Transformer, XGBoost, CNN) and pre-trained models used;
- Complete codes and data that reproduce the results for the study of Out-of-Distribution generalization, the study of the model interpretation and the study of the controlled experiment on ad placement;
- Complete codes and partial data that illustrate how we conduct the study of the In-Silico experiments;
- Complete codes for deploying the web app.

## Computational Requirements
The computation environment is specified by the YAML `environment.yml`. Run the following to set up your environment with conda:
```bash
$ conda env create -f environment.yml
```

For all codes in this repository that reproduce the results in the paper, an Intel CPU on [Colab (an online coding platform provided by Google)](https://colab.google) is sufficient. For codes of Multimodal Large Language Models (MLLM) and Convolutional Neural Networks (CNNs), an NVIDIA T4 GPU with 15GB provided by Colab is sufficient.

## Folder and File Descriptions
### Source Files in `src`
This section describes the source files in the `src` folder:
- `Ad_Gaze_Model`: the folder containing parameters of the 10 XGBoost models for ad gaze predictions.
- `Brand_Gaze_Model`: the folder containing parameters of the 10 XGBoost models for brand gaze predictions.
- `Brand_Share_Model`: the folder containing parameters of the 10 XGBoost models for brand share predictions.
- `CNN_Gaze_Model`: the folder containing parameters of the CNN model fine-tuned by ad gaze (AG), brand gaze (BG) and brand share (BS).
- `EAST-Text-Detection`: the folder containing parameters of the EAST text detection model.
- `Magazine_Topic_Embedding_sample_size15`: the folder containing parameters and configurations of the sentence transformer for topic embeddings.
- `SIFT`: 
    - `kmeans.pt`: the k-mean cluster info used by our SIFT feature extractor, saved in pytorch file.
    - `pca.pt`: the pca info used to compress the SIFT features, saved in pytorch file.
- `Topic_Embedding_PCAs`: the 10 PCA info used to compress the topic embeddings. The ith PCA info was calculated from the ith-fold training data.
- `DL_models.py`: python codes of all deep learning models used in the paper.
- `Predict.py`: python codes of the attention predictions.
- `XGBoost_utils.py`: python codes of all the algorithms and utility functions used.


### Trained Sentence Transformer Model (Note: Faster Performance with GPUs)
Download our [trained model](https://drive.google.com/file/d/1_Vv1AXZsQGw41s-Q3bcg-k6aos5fK0Pd/view?usp=sharing). The downloaded file should be put under `src/Magazine_Topic_Embedding_sample_size15`.

### Preprocessed Data for Model Training
All datasets from the 10-fold-cross-validation experiments are shared in `Shapley_and_ALE_Values/Data`.

## Pretrained XGBoost/CNN Models
All pretrained models are in `src`, categorized by 

## Study: Out-of-Distribution (OOD) Generalization (folder: Out-of-Distribution_Samples)
Run `main.py` to reproduce the OOD generalization results in the paper.

## Study: Model Interpretation (folder: Shapley_and_ALE_Values)
Run the code cells in `Shapley_ALE.ipynb` to reproduce the results of Shapley values and ALE values.

## Study: Controlled Experiment on Ad Placement (folder: Controlled_Experiment_on_Ad_Placement)
The folder includes (1) the raw data that contain the ads (`ads`), the editorials (`eds`) and the table of recorded gaze and ad element sizes (`randomized_experiment_data.sav`).

`data_exp.csv` contains processed ad/brand gaze data given random/observed/no context after matching process. Details about the preprocessing can be found in Section *Controlled Experiment on Ad Placement* of the paper.

Run the R script `R-script_BANOVA_Experiment.txt` on `data_exp.csv` to reproduce the results on this randomized experiment.

## Study: In-Sillico Experiments on Optimal Ad Placement (folder: In-Silico_Experiments_random)
The folder provides codes used by authors to produce the placement optimization results shown in the paper. 

**Note:** Currently, we only share 50 sample ads and their contexts, along with their ground-truth ad and brand gaze, randomly selected from our main dataset to illustrate how the optimization works. These 50 samples are in the subfolder `Sample_Ad_Data/stimuli`, and their ad/brand gaze are in `true_AGs.pt`/`true_BGs.pt` (which should be loaded by `torch.load()`).

The ad placement optimization is done by running the code cells in the `Experiment_Notebook.ipynb`.

## Web App
The web app AdGazer can be initiated by running `AdGazer_WebApp.py`.

**Note:** If GPUs are available locally, for full capability of the AdGazer, you may uncomment the image caption generation code, lines 108-114, and comment out lines 115-117. By default, the image caption generation function is turned off due to potential computation limit.