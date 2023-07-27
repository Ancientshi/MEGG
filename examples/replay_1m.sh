
NET_LIST=(WDL)
SEED_LIST=(0)
DIM=64

for net in ${NET_LIST[@]}
do  
    for seed in ${SEED_LIST[@]}
    do      
        python main.py \
            --BLOCKNUM 15\
            --BASEBLOCK 10\
            --train-batch-size 1000 \
            --test-batch-size 5000 \
            --embedding-dim $DIM \
            --dataset ml-1m \
            --method losschange_replay \
            --net $net \
            --seed $seed \
            --replay_ratio -1 \
            --strategy remain_sides
    
        wait
        echo "seed $seed , All methods are done!"
    done
done

