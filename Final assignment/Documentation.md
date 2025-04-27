# ID
Codalab username: kmilas

# Runnig the main experiment for training the best performing model.
sbatch jobscript_slurm.sh

# Before using Depth Anything.
Besides of setting up the correct model-name in the jobscript you need also to download checkpoints from
https://huggingface.co/depth-anything/Depth-Anything-V2-Base/tree/main
and create a  with name depth/ inside Final Assignment directory where the depth_anything_v2_vitb.pth should be located
# Some Evaluation Code test.py
Select different model_name and different evaluation mode = 'robustness' or mode = '' if not.\
Can also experiment with different checkpoints but be careful of loading the checkpoints and constructing the class of the same model.\
It only works for dinov2b-linear (Segmenter with dinov2_vitb14_reg backbone) without any bugs. \
Do not use multi scale inference as it is not tested well enough leave test_time = '' as it is. \
Run evaluation for one pretrained model with sbatch test.sh

# Low in computing budget
Use sbatch poor_job.sh only when you are out of snellius credits.

