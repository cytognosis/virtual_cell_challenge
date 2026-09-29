#!/bin/bash
# Two-backbone architecture sweep: norman / additive / fold 1, adam, seed 42, deg_sinkhorn_sparse.
#
#   A = model_type origin                    (graph signed recurrent blocks)
#   B = model_type differential_transformer  (scDFM baseline)
#
#   ./submit_architecture_sweep.sh                         submit every arm
#   DRY_RUN=1 ./submit_architecture_sweep.sh               print the sbatch commands only
#   ONLY="A_baseline B_baseline" ./submit_architecture_sweep.sh
#   GAMMA=0.3 ./submit_architecture_sweep.sh
#
# Each arm writes to ${SWEEP_ROOT}/<arm>/. Arm names are <backbone>_<change>.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(pwd)}"
SBATCH_FILE="${SCRIPT_DIR}/train_scdfm.sbatch"

cd "$REPO_DIR" || exit 1
mkdir -p logs

export DATA=norman
export SPLIT=additive
export FOLD=1
export TOPK=30
export NEG=1
export PERT_FN=crispr
export OPTIM=adam
export SEED=42
export ENDPOINT_LOSS=deg_sinkhorn_sparse
export GAMMA="${GAMMA:-0.5}"
export D_MODEL="${D_MODEL:-128}"
export SIGNED=0
export WIRE=0
export NBR_GATE=1
export NOISE=Gaussian
unset THINK_STEPS INFER_TOP COMPILE RESULT_ROOT MODEL_TYPE

SWEEP_ROOT="./result/architecture_sweep/${DATA}_${SPLIT}_f${FOLD}_s${SEED}"
MASK_PATH="data/${DATA}/mask_fold_${FOLD}topk_${TOPK}${SPLIT}_negative_edge.pt"

if [ ! -f "$MASK_PATH" ]; then
  echo "ABORT: mask not found: $MASK_PATH"
  exit 1
fi

# Component sets of each backbone, used for the crossed arms.
COMP_A=(--expression_encoder=log_linear --input_fusion=concat --perturbation_pooling=sum
        --injection=input,condition,decoder,target_node --decoder=gated_mlp --time_scale=1000
        --gene_reinjection=False --vocabulary_interaction=False)
COMP_B=(--expression_encoder=scalar --input_fusion=gene_added --perturbation_pooling=mean_all
        --injection=adapter,decoder --decoder=mlp --time_scale=1.0
        --gene_reinjection=True --vocabulary_interaction=True)

# Value each backbone already uses for a factor; that value is skipped when sweeping it.
declare -A OWN=(
  ["A:expression_encoder"]=log_linear  ["B:expression_encoder"]=scalar
  ["A:input_fusion"]=concat            ["B:input_fusion"]=gene_added
  ["A:perturbation_pooling"]=sum       ["B:perturbation_pooling"]=mean_all
  ["A:decoder"]=gated_mlp              ["B:decoder"]=mlp
  ["A:norm"]=layer                     ["B:norm"]=layer
  ["A:time_scale"]=1000                ["B:time_scale"]=1.0
  ["A:gene_reinjection"]=False         ["B:gene_reinjection"]=True
  ["A:vocabulary_interaction"]=False   ["B:vocabulary_interaction"]=True
  ["A:dropout"]=0.1                    ["B:dropout"]=0.1
  ["A:num_blocks"]=8                   ["B:num_blocks"]=8
  ["A:injection"]=input,condition,decoder,target_node
  ["B:injection"]=adapter,decoder
)

submitted=0

selected() {
  [ -z "${ONLY:-}" ] && return 0
  local name
  for name in $ONLY; do
    [ "$name" = "$1" ] && return 0
  done
  return 1
}

# arm <A|B> <name> [ENV=value ...] -- [run.py flags ...]
arm() {
  local backbone="$1" name="$2"; shift 2
  local model_type=origin
  [ "$backbone" = "B" ] && model_type=differential_transformer

  local arm_name="${backbone}_${name}"
  selected "$arm_name" || return 0

  local exports="ALL,MODEL_TYPE=${model_type}"
  local flags=()
  while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
      shift
      flags=("$@")
      break
    fi
    exports+=",$1"
    shift
  done

  local command=(sbatch --export="$exports" --job-name="arch_${arm_name}" --output="logs/%x_%j.out"
                 "$SBATCH_FILE" --result_path="${SWEEP_ROOT}/${arm_name}" ${flags[@]+"${flags[@]}"})

  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${command[@]}"; printf '\n'
    submitted=$((submitted + 1))
    return 0
  fi
  "${command[@]}" && submitted=$((submitted + 1))
}

# sweep_factor <A|B> <field> <values...>: one arm per value the backbone does not already use.
sweep_factor() {
  local backbone="$1" field="$2"; shift 2
  local value
  for value in "$@"; do
    [ "$value" = "${OWN["${backbone}:${field}"]}" ] && continue
    arm "$backbone" "${field}_${value}" -- "--${field}=${value}"
  done
}

# family <A|B> <label> <block spec>: replace the block family; A keeps a matching refiner, B has none.
family() {
  local backbone="$1" label="$2" spec="$3"
  if [ "$backbone" = "A" ]; then
    arm A "$label" -- --blocks="${spec}*8" --refiner="${spec}"
  else
    arm B "$label" -- --blocks="${spec}:residual_scale=1.0*8" --refiner=none
  fi
}

INJECTIONS=(
  "A_default=input,condition,decoder,target_node"
  "B_default=adapter,decoder"
  "all=input,condition,adapter,decoder,target_node"
  "no_target_node=input,condition,decoder"
  "no_condition=input,decoder,target_node"
  "no_decoder=input,condition,target_node"
)

# References and the block x component crossing
arm A baseline --
arm B baseline --
arm A graph_blocks_B_components -- "${COMP_B[@]}"
arm B diff_blocks_A_components  -- "${COMP_A[@]}"

# Components swept on both backbones
for backbone in A B; do
  sweep_factor "$backbone" expression_encoder log_linear scalar fourier
  sweep_factor "$backbone" input_fusion concat gene_added sum difference
  sweep_factor "$backbone" perturbation_pooling sum mean mean_all
  sweep_factor "$backbone" decoder gated_mlp mlp
  sweep_factor "$backbone" norm layer rms
  sweep_factor "$backbone" time_scale 1000 1.0
  sweep_factor "$backbone" gene_reinjection False True
  sweep_factor "$backbone" vocabulary_interaction False True
  sweep_factor "$backbone" dropout 0.1 0.0
  sweep_factor "$backbone" num_blocks 4 8 12

  for entry in "${INJECTIONS[@]}"; do
    label="${entry%%=*}"
    sites="${entry#*=}"
    [ "$sites" = "${OWN["${backbone}:injection"]}" ] && continue
    arm "$backbone" "injection_${label}" -- --injection="${sites}"
  done

  arm "$backbone" width_256 D_MODEL=256 --
done

# Residual scale, set to the other backbone's value
arm A residual_scale_1.0 -- --residual_scale=1.0
arm B residual_scale_0.1 -- --blocks="differential_transformer:residual_scale=0.1:second_source=control*8"

# Refiner
arm A refiner_none     -- --refiner=none
arm A refine_steps_4   -- --refine_steps=4
arm A refine_steps_16  -- --refine_steps=16
arm A refine_plain     -- --refine_injection=False
arm B refiner_diff       -- --refiner="differential_transformer:residual_scale=1.0:second_source=control"
arm B refiner_diff_plain -- --refiner="differential_transformer:residual_scale=1.0:second_source=control" --refine_injection=False

# Block families on both backbones
family A differential_transformer  differential_transformer
family A differential_stream       "differential_transformer:second_source=stream"
family A transformer               transformer
family A cross_attention           cross_attention
family A latent                    latent
family A mlp                       mlp
family B graph                     graph
family B differential_stream       "differential_transformer:second_source=stream"
family B transformer               transformer
family B cross_attention           cross_attention
family B latent                    latent
family B mlp                       mlp

# Hybrids of the two main block types
arm A hybrid_graph_diff -- --blocks="graph*4,differential_transformer*4"
arm B hybrid_graph_diff -- --blocks="graph:residual_scale=1.0*4,differential_transformer:residual_scale=1.0:second_source=control*4"

# Graph internals and construction on the graph backbone
arm A graph_no_global  -- --blocks="graph:use_global=false*8"      --refiner="graph:use_global=false"
arm A graph_unsigned   -- --blocks="graph:signed_messages=false*8" --refiner="graph:signed_messages=false"
arm A graph_no_source  -- --blocks="graph:use_source=false*8"      --refiner="graph:use_source=false"
arm A graph_no_gate    NBR_GATE=0 --
arm A no_graph         -- --use_graph=False
arm A asymmetric_graph -- --symmetric_graph=False
arm A neighbors_32     -- --max_neighbors=32
arm A neighbors_256    -- --max_neighbors=256
arm A signed_edges     SIGNED=1 --
arm A manifold         WIRE=1 --

# Sampler control on the transformer backbone (neighbor-aware subsets off)
arm B no_graph -- --use_graph=False

echo "submitted ${submitted} arms -> ${SWEEP_ROOT}"