## AdGazer_Replication
This repo provides the codes to reproduce the results from *Jianping Ye, Michel Wedel, Pieters, AdGazer: Improving Contextual Advertising with Theory-Informed Machine Learning*.

## Get Started
Environment Setup:

Run the following to set up your environment with conda:
```bash
$ conda env create -f environment.yml
```

## Pretrained Sentence Transformer Model (Note: Faster Performance with GPUs)
You can download the [pre-trained model](https://drive.google.com/file/d/1_Vv1AXZsQGw41s-Q3bcg-k6aos5fK0Pd/view?usp=sharing). The downloaded file should be put under `src/Magazine_Topic_Embedding_sample_size15`.

## Preprocessed Data for Model Training
All datasets from the 10-fold-cross-validation experiments are shared in `Shapley_and_ALE_Values/Data`.

## Pretrained XGBoost/CNN Models
All pretrained models are in `src`.

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