
NET_LIST=(WDL DCN NFM)
SEED_LIST=(0 1 2 3 4)
DIM=64

for net in ${NET_LIST[@]}
do  
    for seed in ${SEED_LIST[@]}
    do

        METHOD_LIST=(random full_batch fine_tune)
        for METHOD in ${METHOD_LIST[@]}
        do 
            python main.py \
                --BLOCKNUM 15\
                --BASEBLOCK 10\
                --train-batch-size 1000 \
                --test-batch-size 5000 \
                --embedding-dim $DIM \
                --dataset ml-1m \
                --net $net \
                --seed $seed \
                --method $METHOD &
        done
        wait
        
        METHOD_LIST=(herding mir kd)
        for METHOD in ${METHOD_LIST[@]}
        do
            python main.py \
                --BLOCKNUM 15\
                --BASEBLOCK 10\
                --train-batch-size 1000 \
                --test-batch-size 5000 \
                --embedding-dim $DIM \
                --dataset ml-1m \
                --net $net \
                --seed $seed \
                --method $METHOD & 
        done
        wait
        
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
            --strategy remain_sides &
        
        for REPLAY_RATIO in 0.2 0.4 0.6 0.7 0.9 
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
                --replay_ratio $REPLAY_RATIO \
                --strategy remain_sides &
        done
            
        #use all parameters
        python main.py \
            --BLOCKNUM 15\
            --BASEBLOCK 10\
            --train-batch-size 1000 \
            --test-batch-size 5000 \
            --embedding-dim $DIM \
            --dataset ml-1m \
            --method losschange_replay \
            --gradient-fp \
            --net $net \
            --seed $seed \
            --strategy remain_sides

        GGscore_List=(GGscore3)
        for GGscore in ${GGscore_List[@]}
        do
            python main.py \
                --BLOCKNUM 15\
                --BASEBLOCK 10\
                --train-batch-size 1000 \
                --test-batch-size 5000 \
                --embedding-dim $DIM \
                --dataset ml-1m \
                --method $GGscore \
                --net $net \
                --seed $seed \
                --strategy remain_sides &
        done

        losschange_List=(losschange2_replay losschange3_replay losschange4_replay losschange5_replay)
        losschange_List=(losschange2_replay losschange4_replay)
        for losschange in ${losschange_List[@]}
        do
            python main.py \
                --BLOCKNUM 15\
                --BASEBLOCK 10\
                --train-batch-size 1000 \
                --test-batch-size 5000 \
                --embedding-dim $DIM \
                --dataset ml-1m \
                --method $losschange \
                --net $net \
                --seed $seed \
                --strategy remain_sides &
        done


        python main.py \
            --BLOCKNUM 15\
            --BASEBLOCK 10\
            --train-batch-size 1000 \
            --test-batch-size 5000 \
            --embedding-dim $DIM \
            --dataset ml-1m \
            --net $net \
            --seed $seed \
            --method random_with_kd &

        python main.py \
            --BLOCKNUM 15\
            --BASEBLOCK 10\
            --train-batch-size 1000 \
            --test-batch-size 5000 \
            --embedding-dim $DIM \
            --dataset ml-1m \
            --net $net \
            --seed $seed \
            --method herding_with_kd &

        python main.py \
            --BLOCKNUM 15\
            --BASEBLOCK 10\
            --train-batch-size 1000 \
            --test-batch-size 5000 \
            --embedding-dim $DIM \
            --dataset ml-1m \
            --net $net \
            --seed $seed \
            --method mir_with_kd &

        python main.py \
            --BLOCKNUM 15\
            --BASEBLOCK 10\
            --train-batch-size 1000 \
            --test-batch-size 5000 \
            --embedding-dim $DIM \
            --dataset ml-1m \
            --method losschange_replay_with_kd \
            --net $net \
            --seed $seed \
            --strategy remain_sides &

        wait
        echo "seed $seed , All methods are done!"
    done
done

