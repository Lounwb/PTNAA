# Visualization Attribution Map with DMBP (ICCV2021)

## What is DMBP?

The full name of DMBP is `Attribution Map Generation with Disentangled Masked Backpropagation` which was proposed by ["*Attribution Map Generation with Disentangled Masked Backpropagation*"](https://www.arxiv.com/) in [ICCV2021](http://iccv2021.thecvf.com/home).

### How to use?

We have make some modifications to the source code published in `https://gitlab.com/adriaruizo/dmbp_iccv21`. So before run our code, you have to finish setup settings.

#### Setup and Dependencies

- [Python3](https://www.python.org/downloads/)

```shell
conda create -n dmbp python=3.7 anaconda
conda activate dmbp
```

- [PyTorch](http://pytorch.org)

```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

- [dmbp](https://gitlab.com/adriaruizo/dmbp_iccv21)

```shell
git clone https://gitlab.com/adriaruizo/dmbp_iccv21.git
```

Before run the command above, please make sure you have installed git.

#### Get labels of benign and adversarial examples

We provide beginners a python script `get_label.py` to grasp their labels.

Please edit the `line 19` in the script to fit your folder structure. 

```python
image_folder = '../dataset/images'
```

Run the script twice and save the outputs to different files. You will get two label_results, named `classification_results.txt` and `adv_classification_results.txt`.

#### Have your own dmbp visualization results

Run the following command for example:

```shell
python batch_process.py --ori_path ../dataset/images/ --adv_path ../adv/PTNAA/
```

