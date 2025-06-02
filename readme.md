
# Biomarker Prediction from OCT and Clinical Data
![Pipeline Overview](figures/main_arch.png)
*Overview of the biomarker prediction pipeline, showing OCT-only and fusion modes with auxilary reconstruction branch*

A modular, PyTorch-based pipeline for multi-label biomarker detection from medical images (OCT) and clinical data, supporting:

- **Three modes**:  
  - `oct`: Uses only image (OCT) data  
  - `fusion`: Uses both image and clinical data  
  - Optionally, add an image **reconstruction** (multi-task) branch in any mode
- **5-fold cross-validation**, external test set evaluation, TensorBoard logging
- Modular, production-grade code for research and experimentation

---

## Installation

1. **Clone the repository**
    ```
    git clone https://github.com/yourusername/biomarker-fusion.git
    cd biomarker-fusion
    ```



3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

---

## Dataset Preparation

- The dataset must include:
  - OCT image data (single-channel or grayscale)
  - Clinical values: `BCVA`, `CST`
  - Biomarker labels: `B1` to `B6`
- By default, training data is loaded using `load_from_disk()`, and test data with `load_dataset()`.
- Update `--data_path` and `--test_split` in your `main.py` command line as needed.
- To find out more about the data, visit: [Olives](https://github.com/olivesgatech/OLIVES_Dataset)
---

## Running, Training, and Evaluation

### Basic Training (OCT only, no clinical data)
```
python main.py --mode oct --data_path removed_nans_filtered_olives_dataset --test_split olving_test_dataset
```

### Fusion Training (both OCT and clinical data)
```
python main.py --mode fusion --data_path removed_nans_filtered_olives_dataset --test_split olving_test_dataset
```

### Multi Task model (Auxilary Task: Reconstruction)

#### Basic Training including Reconstruction

```
 python main.py --mode oct --reconstruction --data_path removed_nans_filtered_olives_dataset --test_split olving_test_dataset

```
#### Fusion Training including Reconstruction
```
python main.py --mode fusion --reconstruction --data_path removed_nans_filtered_olives_dataset --test_split olving_test_dataset

```

#### Cutsom hyperparameter

```
python main.py --mode fusion --reconstruction --num_epochs 100 --batch_size 128 --recon_weight 0.2 --data_path removed_nans_filtered_olives_dataset --test_split olving_test_dataset

```
## Visualizaiton

you may find the trainig logs/ROC curves and Confusion Matrix in the Run folder results visualization could be seen using command below:

```
tensorboard --logdir="<Run log>"
```
### checkpoints

best models checkpoints are available at: [CHECKPOINTS](https://drive.google.com/drive/folders/12HvGnqXv8A5-ds2ol0ZgOFy6vzwvZf9h?usp=drive_link)

## Citation
if you find something useful about the data, please cite the dataset:

@inproceedings{prabhushankarolives2022,
title={OLIVES Dataset: Ophthalmic Labels for Investigating Visual Eye Semantics},
author={Prabhushankar, Mohit and Kokilepersaud, Kiran and Logan, Yash-yee and Trejo Corona, Stephanie and AlRegib, Ghassan and Wykoff, Charles},
booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 2 (NeurIPS Datasets and Benchmarks 2022) },
year={2022}
}
