#!/bin/bash



# ResNet50 baseline and generation
# Hyperparameters to be aware of:
# lr = 0.1, warmup_epochs = 1, aux_loss_wt = 0.005 for main.py
# lr = 0.1, wd = 0.001 for low_shot.py without generation
# initlr = 1 for train_analogy_generator
# lr = 0.1, wd = 0.001, max_per_label = 5 for low_shot.py with generation


# First train representation
mkdir -p checkpoints/ResNet50_sgm
python ./main.py --model ResNet50_sgm \
  --traincfg base_classes_train_template_smallbatch.yaml \
  --valcfg base_classes_val_template.yaml \
  --print_freq 10 --save_freq 10 \
  --aux_loss_wt 0.005 --aux_loss_type sgm \
  --lr 0.1 --warmup_epochs 1 \
  --checkpoint_dir checkpoints/ResNet50_sgm


# Next save features
mkdir -p features/ResNet50_sgm
python ./save_features.py \
  --cfg train_save_data.yaml \
  --outfile features/ResNet50_sgm/train.hdf5 \
  --modelfile checkpoints/ResNet50_sgm/89.tar \
  --model ResNet50
python ./save_features.py \
  --cfg val_save_data.yaml \
  --outfile features/ResNet50_sgm/val.hdf5 \
  --modelfile checkpoints/ResNet50_sgm/89.tar \
  --model ResNet50_sgm

# Low-shot benchmark without generation
for i in {1..5}
do
  for j in 1 2 5 10 20
  do
    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile features/ResNet50_sgm/train.hdf5 \
      --testfile features/ResNet50_sgm/val.hdf5 \
      --outdir results \
      --lr 0.1 --wd 0.001 \
      --testsetup 1
  done
done

# parse results
echo "ResNet50 results (no generation)"
python ./parse_results.py --resultsdir results \
  --repr ResNet50_sgm \
  --lr 0.1 --wd 0.001


# Train analogy-based generator
mkdir generation
python ./train_analogy_generator.py \
  --lowshotmeta label_idx.json \
  --trainfile features/ResNet50_sgm/train.hdf5 \
  --outdir generation \
  --networkfile checkpoints/ResNet50_sgm/89.tar \
  --initlr 1



# Low-shot benchmark _with_ generation
for i in {1..5}
do
  for j in 1 2 5 10 20
  do
    python ./low_shot.py --lowshotmeta label_idx.json \
      --experimentpath experiment_cfgs/splitfile_{:d}.json \
      --experimentid  $i --lowshotn $j \
      --trainfile features/ResNet50_sgm/train.hdf5 \
      --testfile features/ResNet50_sgm/val.hdf5 \
      --outdir results \
      --lr 0.1 --wd 0.001 \
      --testsetup 1 \
      --max_per_label 5 \
      --generator_name analogies \
      --generator_file generation/ResNet50_sgm/generator.tar
  done
done

# parse results
echo "ResNet50 results (with generation)"
python ./parse_results.py --resultsdir results \
  --repr ResNet50_sgm \
  --lr 0.1 --wd 0.001 \
  --max_per_label 5






