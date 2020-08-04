# Representative-Discriminative Open-set Recognition

This is the implementation of the following paper:

R. Kaviani baghbaderani, Y. Qu. H. Qi, C. Stutts, [Representative-Discriminative Learning for Open-set Land Cover Classification of Satellite Imagery](https://arxiv.org/abs/2007.10891),  European Conference on Computer Vision (ECCV), 2020.

## Pre-requisites
* Python 3.6
* TensorFlow 1.15
* Numpy 1.19
* Scipy 1.5.1
* Scikit-learn 0.23.1

## Dataset
The code uses the following datasets:
1. Pavia University (PaviaU)
2. Pavia Center (Pavia)
3. Indian Pines (Indian_pines)

## Prepare the training dataset
To preprocess the Hyperspcetral data and divide it to Known and Unknown sets:
```python
python preprocessing.py --dataset PaviaU --unk 3 7
```

## Training
To train the network on known set:
```python
python train_rdosr.py --dataset PaviaU
```

## Testing
To test the network on a combination of known and unknown sets:
```python
python test_rdosr.py --dataset PaviaU
```

## Contact
[Razieh Kaviani Baghbaderani](http://web.eecs.utk.edu/~rkavian1/) (rkavian1@vols.utk.edu)
