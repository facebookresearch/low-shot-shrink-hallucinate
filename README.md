# Low-shot Learning by Shrinking and Hallucinating Features

This repository contains code associated with the following paper:<br>
[Low-shot Visual Recognition by Shrinking and Hallucinating Features](https://arxiv.org/abs/1606.02819) <br>
[Bharath Hariharan](http://home.bharathh.info/), [Ross Girshick](http://www.rossgirshick.info/)<br>
arxiv 2016.

You can find trained models [here](https://dl.fbaipublicfiles.com/low-shot-shrink-hallucinate/models.zip).


## Prerequisites
This code uses [pytorch](http://pytorch.org/), [numpy](http://www.numpy.org/) and [h5py](http://www.h5py.org/). It requires GPUs and Cuda.

## Running the code

Running a low-shot learning experiment will involve three or four steps:
1.  Train a ConvNet representation
2.  Save features from the ConvNet
3.  (Optional) Train analogy-based generator
4.  Use saved features to train and test on the low-shot learning benchmark.
Each step is described below.

The scripts directory contains scripts required to generate results for the baseline representation, representations trained with the SGM loss or L2 regularization, and results with and without the analogy-based generation strategy.


### Training a ConvNet representation
To train the ConvNet, we first need to specify the training and validation sets.
The training and validation datasets, together with data-augmentation and preprocessing steps, are specified through yaml files: see `base_classes_train_template.yaml` and `base_classes_val_template.yaml`.
You will need to specify the path to the directory containing ImageNet in each file.

The main entry point for training a ConvNet representation is `main.py`. For example, to train a ResNet10 representation with the sgm loss, run:

    mkdir -p checkpoints/ResNet10_sgm
    python ./main.py --model ResNet10 \
      --traincfg base_classes_train_template.yaml \
      --valcfg base_classes_val_template.yaml \
      --print_freq 10 --save_freq 10 \
      --aux_loss_wt 0.02 --aux_loss_type sgm \
      --checkpoint_dir checkpoints/ResNet10_sgm
Here, `aux_loss_type` is the kind of auxilliary loss to use (`sgm` or `l2` or `batchsgm`), `aux_loss_wt` is the weight attached to this auxilliary loss, and `checkpoint_dir` is a cache directory to save the checkpoints.

The model checkpoints will be saved as epoch-number.tar. Training by default runs for 90 epochs, so the final model saved will be `89.tar`.

### Saving features from the ConvNet
The next step is to save features from the trained ConvNet. This is fairly straightforward: first, create a directoryto save the features in, and then save the features for the train set and the validation set. Thus, for the ResNet10 model trained above:

    mkdir -p features/ResNet10_sgm
    python ./save_features.py \
      --cfg train_save_data.yaml \
      --outfile features/ResNet10_sgm/train.hdf5 \
      --modelfile checkpoints/ResNet10_sgm/89.tar \
      --model ResNet10
    python ./save_features.py \
      --cfg val_save_data.yaml \
      --outfile features/ResNet10_sgm/val.hdf5 \
      --modelfile checkpoints/ResNet10_sgm/89.tar \
      --model ResNet10


### Training the analogy-based generator
The entry point for training the analogy-based generator is `train_analogy_generator.py`.
To train the analogy based generation on the above representation, run:

    mkdir generation
    python ./train_analogy_generator.py \
      --lowshotmeta label_idx.json \
      --trainfile features/ResNet10_sgm/train.hdf5 \
      --outdir generation \
      --networkfile checkpoints/ResNet10_sgm/89.tar \
      --initlr 1

Here, label_idx.json contains the split of base and novel classes, and is used to pick out the saved features corresponding to just the base classes.
The analogy generation has several steps and maintains a cache.
The final generator will be saved in generation/ResNet10_sgm/generator.tar

### Running the low shot benchmark
The benchmark tests with 5 different settings for the number of novel category examples
_n_ = {1,2,5,10,20}.
The benchmark is organized into 5 experiments, with each experiment corresponding to a fixed choice of _n_ examples for each category.

The main entry point for running the low shot benchmark is `low_shot.py`, which will run a single experiment for a single value of _n_. Thus, to run the benchmark, `low_shot.py` will have to be run 25 times. This design choice has been made to allow the 25 experiments to be run in parallel.

There is one final wrinkle. To allow cross-validation of hyperparameters, there are two different setups for the benchmark: a validation setup, and a test setup. The setups use different settings for the hyperparameters.

To run the benchmark, first create a results directory, and then run each experiment for each value of _n_. For example, running the first experiment with _n_=2 on the test setup will look like:

    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  1 --lowshotn 2 \
      --trainfile features/ResNet10_sgm/train.hdf5 \
      --testfile features/ResNet10_sgm/val.hdf5 \
      --outdir results \
      --lr 1 --wd 0.001 \
      --testsetup 1

If you want to use the analogy based generator, and generate till there are at least 5 examples per category, then you can run:

    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  1 --lowshotn 2 \
      --trainfile features/ResNet10_sgm/train.hdf5 \
      --testfile features/ResNet10_sgm/val.hdf5 \
      --outdir results \
      --lr 1 --wd 0.001 \
      --testsetup 1 \
      --max_per_label 5 \
      --generator_name analogies \
      --generator_file generation/ResNet10_sgm/generator.tar

Here `generator_name` is the kind of generator to use; only analogy based generation is implemented, but other ways of generating data can easily be added (see below).

Once all the experiments are done, you can use the quick-and-dirty script `parse_results.py` to assemble the results:

    python ./parse_results.py --resultsdir results \
      --repr ResNet10_sgm \
      --lr 1 --wd 0.001 \
      --max_per_label 5

## Extensions

### New losses
It is fairly easy to implement novel loss functions or forms of regularization. Such losses can be added to `losses.py`, and can make use of the scores, the features, and even the model weights. Create your own loss function, add it to the dictionary of auxiliary losses in `GenericLoss`, and specify how it should be called in the `__call__` function.

### New generation strategies
Again, implementing new data generation strategies is also easy. Any generation strategy should provide two functions:

  1. `init_generator` should be able to load whatever state you need to load from a single filename provided as input and return a generator.
  2. `do_generate` should take four arguments: the original set of novel class feats, novel class labels, the generator produced by `init_generator` and the total number of examples per class we want to target. It should return a new set of novel class feats and novel class labels that include both the real and the generated examples.

Add any new generation strategy to `generation.py`.

## Matching Networks
This repository also includes an implementation of [Matching Networks](https://arxiv.org/abs/1606.04080). Given a saved feature representation (such as the one above), you can train matching networks by running:

    python matching_network.py --test 0 \
      --trainfile features/ResNet10_sgm/train.hdf5 \
      --lowshotmeta label_idx.json \
      --modelfile matching_network_sgm.tar

This will save the trained model in `matching_network_sgm.tar`.
Then, test the model using:

    python matching_network.py --test 1 \
      --trainfile features/ResNet10_sgm/train.hdf5 \
      --testfile features/ResNet10_sgm/val.hdf5 \
      --lowshotmeta label_idx.json \
      --modelfile matching_network_sgm.tar \
      --lowshotn 1 --experimentid 1 \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --outdir results

As in the benchmark above, this tests a single experiment for a single value of _n_.

## New results
The initial implementation, corresponding to the original paper, was in Lua. For this release, we have switched to Pytorch. As such, there are small differences in the resulting numbers, although the trends are the same. The new numbers are below:

_Top-1, Novel classes_

| Representation      |	Low-shot phase     | 	n=1 | 	2   | 	5   | 	10   | 	20   |
| :------------	      |  :-------------    | :----: | :---: | :---: | :----: | :---: |
| Baseline	          |  Baseline          |  2.77  | 10.78 | 26.38 | 35.46	 | 41.49 |
| Baseline	          |  Generation	       |  9.17	| 15.85	| 25.47	| 33.21	 | 40.41 |
| SGM	              |  Baseline	       |  4.14	| 13.08	| 27.83	| 36.04	 | 41.36 |
| SGM	              |  Generation	       |  9.85	| 17.32	| 27.89	| 36.17	 | 41.42 |
| Batch SGM	          |  Baseline	       |  4.16	| 13.01	| 28.12	| 36.56	 | 42.07 |
| L2	              |  Baseline	       |  7.14	| 16.75	| 27.73	| 32.32	 | 35.11 |
| Baseline	          |  Matching Networks |  18.33	| 23.87	| 31.08	| 35.27	 | 38.45 |
| Baseline (Resnet 50)|  Baseline          |  6.82  | 18.37 | 36.55 | 46.15  | 51.99 |
| Baseline (Resnet 50)|  Generation	       |  16.58	| 25.38	| 36.16	| 44.53	 | 52.06 |
| SGM (Resnet 50)	  |  Baseline	       |  10.23	| 21.45	| 37.25	| 46.00	 | 51.83 |
| SGM (Resnet 50)     |  Generation	       |  15.77	| 24.43	| 37.22	| 45.96	 | 51.82 |


_Top-5, Novel classes_

| Representation      |	Low-shot phase     | 	n=1 | 	2   | 	5   | 	10   | 	20   |
| :------------	      |  :-------------    | :----: | :---: | :---: | :----: | :---: |
| Baseline	          |  Baseline          | 14.10	| 33.34	| 56.20	| 66.15	 | 71.52 |
| Baseline	          |  Generation	       | 29.68	| 42.15	| 56.13	| 64.52	 | 70.56 |
| SGM	              |  Baseline	       | 23.14	| 42.37	| 61.68	| 69.60	 | 73.76 |
| SGM	              |  Generation	       | 32.80	| 46.37	| 61.70	| 69.71	 | 73.81 |
| Batch SGM	          |  Baseline	       | 22.97	| 42.35	| 61.91	| 69.91	 | 74.45 |
| L2	              |  Baseline	       | 29.08	| 47.42	| 62.33	| 67.96	 | 70.63 |
| Baseline	          |  Matching Networks | 41.27	| 51.25	| 62.13	| 67.82	 | 71.78 |
| Baseline (Resnet 50)|  Baseline          | 28.16	| 51.03	| 71.01	| 78.39	 | 82.32 |
| Baseline (Resnet 50)|  Generation	       | 44.76	| 58.98	| 71.37	| 77.65	 | 82.30 |
| SGM (Resnet 50)	  |  Baseline	       | 37.81	| 57.08	| 72.78	| 79.09	 | 82.61 |
| SGM (Resnet 50)     |  Generation	       | 45.11	| 58.83	| 72.76	| 79.09	 | 82.61 |

_Top-1, Base classes_

| Representation      |	Low-shot phase     | 	n=1 | 	2   | 	5   | 	10   | 	20   |
| :------------	      |  :-------------    | :----: | :---: | :---: | :----: | :---: |
| Baseline	          |  Baseline          | 71.04	| 69.63	| 65.67	| 63.56	 | 62.83 |
| Baseline	          |  Generation	       | 72.38	| 70.12	| 68.50	| 68.11	 | 69.47 |
| SGM	              |  Baseline	       | 75.76	| 74.24	| 70.82	| 69.02	 | 68.29 |
| SGM	              |  Generation	       | 72.62	| 71.05	| 70.86	| 68.88	 | 68.24 |
| Batch SGM	          |  Baseline	       | 75.75	| 74.50	| 70.83	| 68.87	 | 68.04 |
| L2	              |  Baseline	       | 74.50	| 72.26	| 69.99	| 69.62	 | 69.42 |
| Baseline	          |  Matching Networks | 48.71	| 52.10	| 58.65	| 62.55	 | 65.25 |
| Baseline (Resnet 50)|  Baseline          | 83.16	| 81.94	| 78.36	| 76.27	 | 75.32 |
| Baseline (Resnet 50)|  Generation	       | 79.39	| 77.81	| 76.86	| 76.12	 | 75.27 |
| SGM (Resnet 50)	  |  Baseline	       | 83.96	| 82.52	| 79.04	| 76.78	 | 75.37 |
| SGM (Resnet 50)     |  Generation	       | 81.17	| 79.60	| 79.04	| 76.84	 | 75.35 |

_Top-5, Base classes_

| Representation      |	Low-shot phase     | 	n=1 | 	2   | 	5   | 	10   | 	20   |
| :------------	      |  :-------------    | :----: | :---: | :---: | :----: | :---: |
| Baseline	          |  Baseline          | 88.90	| 87.53	| 84.56	| 83.23	 | 82.76 |
| Baseline	          |  Generation	       | 88.32	| 86.81	| 85.61	| 85.56	 | 86.97 |
| SGM	              |  Baseline	       | 91.00	| 89.32	| 86.67	| 85.51	 | 84.97 |
| SGM	              |  Generation	       | 88.43	| 87.12	| 86.62	| 85.49	 | 84.95 |
| Batch SGM	          |  Baseline	       | 91.13	| 89.35	| 86.55	| 85.38	 | 84.88 |
| L2	              |  Baseline	       | 90.03	| 87.84	| 85.99	| 85.63	 | 85.60 |
| Baseline	          |  Matching Networks | 76.70	| 77.82	| 80.59	| 82.19	 | 83.27 |
| Baseline (Resnet 50)|  Baseline          | 95.11	| 94.02	| 91.84	| 90.71	 | 90.23 |
| Baseline (Resnet 50)|  Generation	       | 92.30	| 91.26	| 90.48	| 90.33	 | 90.18 |
| SGM (Resnet 50)	  |  Baseline	       | 95.25	| 93.81	| 91.32	| 89.89	 | 89.28 |
| SGM (Resnet 50)     |  Generation	       | 92.94	| 91.67	| 91.35	| 89.90	 | 89.21 |

_Top-1, All classes_

| Representation      |	Low-shot phase     | 	n=1 | 	2   | 	5   | 	10   | 	20   |
| :------------	      |  :-------------    | :----: | :---: | :---: | :----: | :---: |
| Baseline	          |  Baseline          | 29.16	| 33.53	| 41.57	| 46.33	 | 49.74 |
| Baseline	          |  Generation	       | 33.60	| 36.83	| 42.11	| 46.70	 | 51.65 |
| SGM	              |  Baseline	       | 31.83	| 36.73	| 44.45	| 48.79	 | 51.77 |
| SGM	              |  Generation	       | 34.12	| 38.09	| 44.50	| 48.82	 | 51.79 |
| Batch SGM	          |  Baseline	       | 31.84	| 36.78	| 44.63	| 49.05	 | 52.11 |
| L2	              |  Baseline	       | 33.18	| 38.21	| 44.07	| 46.74	 | 48.37 |
| Baseline	          |  Matching Networks | 30.08	| 34.78	| 41.74	| 45.81	 | 48.81 |
| Baseline (Resnet 50)|  Baseline          | 36.33	| 42.95	| 52.71	| 57.79	 | 61.01 |
| Baseline (Resnet 50)|  Generation	       | 40.86	| 45.65	| 51.90	| 56.74	 | 61.03 |
| SGM (Resnet 50)	  |  Baseline	       | 38.73	| 45.06	| 53.40	| 57.90	 | 60.93 |
| SGM (Resnet 50)     |  Generation	       | 41.05	| 45.76	| 53.39	| 57.90	 | 60.92 |

_Top-5, All classes_

| Representation      |	Low-shot phase     | 	n=1 | 	2   | 	5   | 	10   | 	20   |
| :------------	      |  :-------------    | :----: | :---: | :---: | :----: | :---: |
| Baseline	          |  Baseline          | 43.02	| 54.29	| 67.17	| 72.75	 | 75.86 |
| Baseline	          |  Generation	       | 52.35	| 59.42	| 67.53	| 72.65	 | 76.91 |
| SGM	              |  Baseline	       | 49.37	| 60.52	| 71.34	| 75.75	 | 78.10 |
| SGM	              |  Generation	       | 54.31	| 62.12	| 71.33	| 75.81	 | 78.12 |
| Batch SGM	          |  Baseline	       | 49.32	| 60.52	| 71.44	| 75.89	 | 78.48 |
| L2	              |  Baseline	       | 52.65	| 63.05	| 71.48	| 74.79	 | 76.41 |
| Baseline	          |  Matching Networks | 54.97	| 61.52	| 69.27	| 73.38	 | 76.22 |
| Baseline (Resnet 50)|  Baseline          | 54.05	| 67.65	| 79.07	| 83.15	 | 85.37 |
| Baseline (Resnet 50)|  Generation	       | 63.14	| 71.45	| 78.76	| 82.55	 | 85.35 |
| SGM (Resnet 50)	  |  Baseline	       | 60.02	| 71.28	| 79.95	| 83.27	 | 85.19 |
| SGM (Resnet 50)     |  Generation	       | 63.60	| 71.53	| 79.95	| 83.27	 | 85.16 |
