WANDB_PROJECT=verbs
TAG=main

python run_verbs.py --wandb_project $WANDB_PROJECT --dataset_path 'src/verbs/dataset.pkl' --exp_tag $TAG 