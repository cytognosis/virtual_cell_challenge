#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=verma198@purdue.edu
#SBATCH --mail-type=FAIL

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

module load conda
conda activate /depot/natallah/data/shourya/scDFM_env

cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH=./
export SCDFM_COMPILE="${COMPILE:-0}"

# Configuration comes from environment variables with the defaults below. Any flag passed
# on the command line (sbatch train_scdfm.sbatch --blocks=... --lr=...) overrides the
# corresponding default and is forwarded to run.py unchanged.
DATA="${DATA:-combosciplex}"
SPLIT="${SPLIT:-additive}"
FOLD="${FOLD:-0}"
TOPK="${TOPK:-30}"
NEG="${NEG:-1}"
SIGNED="${SIGNED:-0}"
WIRE="${WIRE:-0}"
NBR_GATE="${NBR_GATE:-1}"
NOISE="${NOISE:-Gaussian}"
INFER_TOP="${INFER_TOP:-1000}"
D_MODEL="${D_MODEL:-128}"

case "${PERT_FN:-}" in
  drug) PERT=label ;;
  crisper|crispr) PERT=crispr ;;
  *) case "$DATA" in
       norman*) PERT=crispr ;;
       *) PERT=label ;;
     esac ;;
esac

USER_ARGS=("$@")
ARGS=()

provided() {
  local name="${1//-/_}" arg key
  for arg in "${USER_ARGS[@]}"; do
    key="${arg%%=*}"
    key="${key#--}"
    key="${key//-/_}"
    if [[ "$key" == "$name" || "$key" == "no_$name" ]]; then
      return 0
    fi
  done
  return 1
}

default_arg() {
  provided "$1" || ARGS+=("--$1=$2")
}

# switch <field> <config default 0|1> <requested 0|1>: emits a flag only when it differs from the default.
switch() {
  local field="$1" default="$2" want="$3"
  provided "$field" && return 0
  [ "$want" = "$default" ] && return 0
  if [ "$want" = "1" ]; then
    ARGS+=("--${field//_/-}")
  else
    ARGS+=("--no-${field//_/-}")
  fi
}

default_arg data_name "$DATA"
default_arg split_method "$SPLIT"
default_arg fold "$FOLD"
default_arg topk "$TOPK"
default_arg perturbation_type "$PERT"
default_arg mode velocity
default_arg noise_type "$NOISE"
switch use_negative_edge 0 "$NEG"
switch use_signed_edges 0 "$SIGNED"
switch use_manifold 0 "$WIRE"
switch neighbor_gate 1 "$NBR_GATE"

default_arg model_type "${MODEL_TYPE:-origin}"
default_arg d_model "$D_MODEL"
default_arg d_hid "${D_HID:-$((4 * D_MODEL))}"
default_arg infer_top_gene "$INFER_TOP"
default_arg n_top_genes "${N_TOP_GENES:-5000}"
if [ -n "${THINK_STEPS:-}" ]; then
  default_arg refine_steps "$THINK_STEPS"
fi

default_arg optimizer "${OPTIM:-adam}"
default_arg batch_size "${BATCH_SIZE:-48}"
default_arg lr "${LR:-5e-5}"
default_arg eta_min "${ETA_MIN:-1e-6}"
default_arg steps "${STEPS:-200000}"
default_arg seed "${SEED:-42}"

default_arg endpoint_loss "${ENDPOINT_LOSS:-mmd}"
default_arg gamma "${GAMMA:-0.5}"

default_arg result_path "${RESULT_ROOT:-./result/${DATA}_${SPLIT}_f${FOLD}}"

printf 'python src/script/run.py'
printf ' %q' "${ARGS[@]}" "$@"
printf '\n'

python src/script/run.py "${ARGS[@]}" "$@"
status=$?

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff=$(($ends-$starts))
hours=$(($diff / 3600))
dif=$(($diff % 3600))
minutes=$(($dif / 60))
seconds=$(($dif % 60))

printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
printf "\t================================\n\n"

sacct --jobs=$SLURM_JOBID --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed

exit $status