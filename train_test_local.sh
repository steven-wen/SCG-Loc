#config="MixCvT13_DPCGFPN128_outstride1nearest_CE_Balance_cr33.py"
config="${1:-DPCG+dsk_0.84.py}"

set -e

gpu_ids=0
python_bin="/home/zbw/miniconda3/envs/zbw/bin/python"
project_root="$(pwd)"
config_base="$(basename "$config" .py)"
timestamp="$(date +%Y%m%d_%H%M%S)"
name="${config_base}/${timestamp}"

cleanup_cache() {
    find "$project_root" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
    find "$project_root" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
}

trap cleanup_cache EXIT

cleanup_cache
CUDA_VISIBLE_DEVICES=$gpu_ids "$python_bin" train.py --config "$config" --name "$name"
cleanup_cache

cd "checkpoints/$name" || exit 1
test_config="$(basename "$config")"
echo "$test_config"
cleanup_cache
CUDA_VISIBLE_DEVICES=$gpu_ids "$python_bin" test_meter.py --config "$test_config"
cleanup_cache
