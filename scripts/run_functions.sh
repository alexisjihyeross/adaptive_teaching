WANDB_PROJECT=functions

PROG_DIR='src/programs/functions/target_concepts'
TAG=main

for ((i=0; i<24; i++))
do
	for EXP_ID in synthetic 
	do
		python run_functions.py \
			--exp_id ${EXP_ID} \
			--wandb_project ${WANDB_PROJECT} \
			--prog_concept_params_file ${PROG_DIR}/${i}.json \
            --outputs_by_inp_file tuple_progs_to_outputs_by_inp.json \
			--exp_tag ${TAG} 

	done
done
