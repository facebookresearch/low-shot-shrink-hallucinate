#!/bin/bash



# ResNet10 with SGM loss and generation
# Hyperparameters to be aware of:
# aux_loss_wt = 0.03 for main.py
# lr = 0.1, wd = 0.001 for low_shot.py without generation


# First train representation
mkdir -p checkpoints/ResNet10_l2
python ./main.py --model ResNet10 \
  --traincfg base_classes_train_template.yaml \
  --valcfg base_classes_val_template.yaml \
  --print_freq 10 --save_freq 10 \
  --aux_loss_wt 0.03 --aux_loss_type l2 \
  --checkpoint_dir checkpoints/ResNet10_l2


# Next save features
mkdir -p features/ResNet10_l2
python ./save_features.py \
  --cfg train_save_data.yaml \
  --outfile features/ResNet10_l2/train.hdf5 \
  --modelfile checkpoints/ResNet10_l2/89.tar \
  --model ResNet10
python ./save_features.py \
  --cfg val_save_data.yaml \
  --outfile features/ResNet10_l2/val.hdf5 \
  --modelfile checkpoints/ResNet10_l2/89.tar \
  --model ResNet10

# Low-shot benchmark without generation
for i in {1..5}
do
  for j in 1 2 5 10 20
  do
    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile features/ResNet10_l2/train.hdf5 \
      --testfile features/ResNet10_l2/val.hdf5 \
      --outdir results \
      --lr 0.1 --wd 0.001 \
      --testsetup 1
  done
done

# parse results
echo "ResNet10 SGM results (no generation)"
python ./parse_results.py --resultsdir results \
  --repr ResNet10_l2 \
  --lr 0.1 --wd 0.001


