#!/bin/bash
# 文件名: tune_kmeans.sh

INPUT_DIR="Data/轮次分类CSV"
EMB_MODEL="/Users/vince/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1"
OUTPUT_DIR="./cluster_results"
VISUALIZE="--visualize"
EVAL="--eval"

mkdir -p $OUTPUT_DIR

BEST_SCORE=-1
BEST_K=0

for K in {2..10}; do
    OUT_FILE="${OUTPUT_DIR}/kmeans_k${K}.csv"

    echo ">>> 测试 KMeans n_clusters=${K} ..."
    SCORE=$(python code/cluster_dialog.py \
        --input $INPUT_DIR \
        --output $OUT_FILE \
        --algorithm kmeans \
        --n_clusters $K \
        --embedding_model $EMB_MODEL \
        --embedding_mode qa_turn \
        --normalize \
        $VISUALIZE \
        $EVAL 2>&1 | grep "Silhouette Score" | awk '{print $3}')

    echo "n_clusters=${K}, Silhouette Score=${SCORE}"

    # 比较最优分数
    if [[ ! -z "$SCORE" ]]; then
        SCORE_VAL=$(echo $SCORE | sed 's/[^0-9\.\-]//g')
        if (( $(echo "$SCORE_VAL > $BEST_SCORE" | bc -l) )); then
            BEST_SCORE=$SCORE_VAL
            BEST_K=$K
        fi
    fi
done

echo "===================================="
echo "最优 K = $BEST_K, 对应 Silhouette Score = $BEST_SCORE"
echo "===================================="