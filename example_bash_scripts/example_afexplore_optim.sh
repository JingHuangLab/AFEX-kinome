#!/bin/bash
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p gpu
#SBATCH -J AFEX
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

echo "Running on node: $SLURM_JOB_NODELIST"

# ----------- User-defined variables -----------
protein_name='EGFR'
target_state="active"
protein_type="kinase"
nclust=256 # Number of MSA clusters used for featurization
nsteps=1000 # Maximum iterations
nsuccess=50 # Required number of successful active-state samplings
learning_rate=0.02
plddt_target=0.95 # Minimum pLDDT threshold for accepted structures

# ----------- Path configuration -----------
work_path="/home/usr/projects/kinase/AFEX" # Project working directory (modify as needed)
afparam_dir="/home/usr/projects/kinase/alphafold2" # Location of AlphaFold2 model parameters (user-supplied, check https://github.com/google-deepmind/alphafold/tree/main)
rawfeat_path="$afparam_dir/$protein_name" # Directory with AlphaFold2 raw features for target protein (features.pkl)
output_dir="/home/usr/projects/kinase/AFEX/results_lr${learning_rate}_plddt${plddt_target}/$protein_name/${target_state}_state"

# ----------- Execution -----------
# Run the main Python optimization script with specified parameters
python "$work_path/afexplore_optim_active5_type2.py" \
       --rawfeat_dir "$rawfeat_path" \
       --output_dir "$output_dir" \
       --afparam_dir "$afparam_dir" \
       --nclust "$nclust" \
       --nsteps "$nsteps" \
       --target_state "$target_state" \
       --num_success "$nsuccess" \
       --protein_type "$protein_type" \
       --learning_rate "$learning_rate" \
       --plddt_target "$plddt_target"

# ----------- Job status check -----------
# If Python script fails, exit with error
if [ $? -ne 0 ]; then
  echo "Error: Python script failed."
  exit 1
fi

