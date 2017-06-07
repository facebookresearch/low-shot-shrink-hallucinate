#!/bin/bash



# ResNet10 baseline and generation
# Hyperparameters to be aware of:
# lr = 0.1, wd = 0.01 for low_shot.py without generation
# lr = 0.1, wd = 0.001, max_per_label = 20 for low_shot.py with generation


# First train representation
mkdir -p checkpoints/ResNet10
python ./main.py --model ResNet10 \
  --traincfg base_classes_train_template.yaml \
  --valcfg base_classes_val_template.yaml \
  --print_freq 10 --save_freq 10 \
  --aux_loss_wt 0 \
  --checkpoint_dir checkpoints/ResNet10


# Next save features
mkdir -p features/ResNet10
python ./save_features.py \
  --cfg train_save_data.yaml \
  --outfile features/ResNet10/train.hdf5 \
  --modelfile checkpoints/ResNet10/89.tar \
  --model ResNet10
python ./save_features.py \
  --cfg val_save_data.yaml \
  --outfile features/ResNet10/val.hdf5 \
  --modelfile checkpoints/ResNet10/89.tar \
  --model ResNet10

# Low-shot benchmark without generation
for i in {1..5}
do
  for j in 1 2 5 10 20
  do
    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile features/ResNet10/train.hdf5 \
      --testfile features/ResNet10/val.hdf5 \
      --outdir results \
      --lr 0.1 --wd 0.01 \
      --testsetup 1
  done
done

# parse results
echo "ResNet10 results (no generation)"
python ./parse_results.py --resultsdir results \
  --repr ResNet10 \
  --lr 0.1 --wd 0.01


# Train analogy-based generator
mkdir generation
python ./train_analogy_generator.py \
  --lowshotmeta label_idx.json \
  --trainfile features/ResNet10/train.hdf5 \
  --outdir generation \
  --networkfile checkpoints/ResNet10/89.tar



# Low-shot benchmark _with_ generation
for i in {1..5}
do
  for j in 1 2 5 10 20
  do
    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile features/ResNet10/train.hdf5 \
      --testfile features/ResNet10/val.hdf5 \
      --outdir results \
      --lr 0.1 --wd 0.001 \
      --testsetup 1 \
      --max_per_label 20 \
      --generator_name analogies \
      --generator_file generation/ResNet10/generator.tar
  done
done

# parse results
echo "ResNet10 results (with generation)"
python ./parse_results.py --resultsdir results \
  --repr ResNet10 \
  --lr 0.1 --wd 0.001 \
  --max_per_label 20






