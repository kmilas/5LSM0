# Runnig the main experiment for training the best performing model.
sbatch jobscript_slurm.sh
# Training the other models proposed in the paper.
Change model-name in jobscript_slurm.sh and use the specific training settings reported in the paper.
# Some Evaluation Code test.py
Select different model_name and different evaluation mode = 'robustness' or mode = '' if not.\
Can also experiment with different checkpoints but be careful of loading the checkpoints and constructing the class of the same model.\
Do not use multi scale inference as it is not tested well enough
Leave test_time = '' as it is.
Run evaluation for one pretrained model with sbatch test.sh

# Low in computing budget
Use sbatch poor_job.sh only when you are out of snellius credits.
