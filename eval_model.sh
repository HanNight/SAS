set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime")
OUTPUT_DIR="$HOME"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-tokens)
            NUM_TOKENS="$2"
            if [ "$NUM_TOKENS" -lt 32768 ]; then
                MAX_TOKENS=32768
            else
                MAX_TOKENS=$((NUM_TOKENS * 2))
            fi
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of Tokens: ${NUM_TOKENS}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_gen \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}_${NUM_TOKENS}.parquet \
        data.n_samples=16 \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=${NUM_TOKENS} \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1 \
        rollout.max_num_batched_tokens=${MAX_TOKENS} \
        rollout.free_cache_engine=False \
        rollout.name=vllm
done