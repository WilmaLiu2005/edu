#!/bin/bash

INPUT_DIR="Data/轮次分类CSV"
EMB_MODEL="/Users/vince/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1"
OUTPUT_DIR="./cluster_results"
VISUALIZE="--visualize"
EVAL="--eval"

mkdir -p $OUTPUT_DIR

# ---------------------------
# DBSCAN 超参数搜索
# ---------------------------
echo "eps,min_samples,silhouette_score" > "$OUTPUT_DIR/dbscan_results.csv"
BEST_SCORE=-1
BEST_PARAMS=""
for EPS in 0.2 0.3 0.4 0.5
do
    for MIN_SAMPLES in 2 3 5
    do
        OUT_FILE="$OUTPUT_DIR/dbscan_eps${EPS}_min${MIN_SAMPLES}.csv"
        echo "DBSCAN eps=$EPS min_samples=$MIN_SAMPLES"
        SCORE=$(python /Users/vince/undergraduate/KEG/edu/code/cluster_dialog.py \
            --input $INPUT_DIR \
            --output $OUT_FILE \
            --algorithm dbscan \
            --eps $EPS \
            --min_samples $MIN_SAMPLES \
            --embedding_model $EMB_MODEL \
            --embedding_mode qa_turn \
            --normalize \
            $EVAL $VISUALIZE 2>&1 | grep "Silhouette Score" | awk '{print $3}')
        
        # 保存到 CSV
        echo "$EPS,$MIN_SAMPLES,$SCORE" >> "$OUTPUT_DIR/dbscan_results.csv"
        
        # 比较更新最优
        if (( $(echo "$SCORE > $BEST_SCORE" | bc -l) )); then
            BEST_SCORE=$SCORE
            BEST_PARAMS="eps=$EPS min_samples=$MIN_SAMPLES"
        fi
    done
done
echo "Best DBSCAN Silhouette Score: $BEST_SCORE with $BEST_PARAMS"

# ---------------------------
# HDBSCAN 超参数搜索
# ---------------------------
echo "min_cluster_size,metric,silhouette_score" > "$OUTPUT_DIR/hdbscan_results.csv"
BEST_SCORE=-1
BEST_PARAMS=""
for MIN_CLUSTER in 2 3 5
do
    for METRIC in cosine euclidean
    do
        OUT_FILE="$OUTPUT_DIR/hdbscan_mc${MIN_CLUSTER}_metric${METRIC}.csv"
        echo "HDBSCAN min_cluster_size=$MIN_CLUSTER metric=$METRIC"
        SCORE=$(python cluster_dialog.py \
            --input $INPUT_DIR \
            --output $OUT_FILE \
            --algorithm hdbscan \
            --min_cluster_size $MIN_CLUSTER \
            --metric $METRIC \
            --embedding_model $EMB_MODEL \
            --embedding_mode qa_turn \
            --normalize \
            $EVAL $VISUALIZE 2>&1 | grep "Silhouette Score" | awk '{print $3}')
        
        # 保存到 CSV
        echo "$MIN_CLUSTER,$METRIC,$SCORE" >> "$OUTPUT_DIR/hdbscan_results.csv"
        
        if (( $(echo "$SCORE > $BEST_SCORE" | bc -l) )); then
            BEST_SCORE=$SCORE
            BEST_PARAMS="min_cluster_size=$MIN_CLUSTER metric=$METRIC"
        fi
    done
done
echo "Best HDBSCAN Silhouette Score: $BEST_SCORE with $BEST_PARAMS"
