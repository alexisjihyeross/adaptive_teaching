WANDB_PROJECT=fractions

TAG=main

python run_fractions.py \
	--wandb_project ${WANDB_PROJECT} \
	--outputs_by_inp_file frac_tuple_progs_to_outputs_by_inp.json \
	--exp_tag ${TAG} 