#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=4
export MASTER_PORT=29571
export CPUS_PER_TASK=96
export QUOTA=reserved
export LAM=0.0
export LR=8e-5
export SSTEP=200
export MSTEP=50
export QUANT=1
while true ; do
    PYTHONPATH="$(dirname $0)/":$PYTHONPATH \
    srun -p Your_Partition \
        --nodes=$NNODES\
        --ntasks-per-node=1 \
        --gres=gpu:$GPUS_PER_NODE \
        --cpus-per-task=$CPUS_PER_TASK \
        --kill-on-bad-exit=1 \
        --quotatype=$QUOTA \
        bash -c 'torchrun \
        --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} \
        train_scripts/train_tri.py \
        config/PixArt_xl2_img512_internal_for_3d_sample_training_long.py \
        --work-dir output/trained_model_zpp_512_e30_190x2kx${QUANT}_570k_n_stren_lr=${LR}_32x32_lam=${LAM}_all_step_lim=${MSTEP}_syth_lim=${SSTEP}_tri_long \
        --report_to=tensorboard \
        --loss_report_name=train_loss \
        --lr=${LR} \
        --mix_ratio=1 \
        --lam=${LAM} \
        --min_syth_step=${SSTEP} \
        --all_3d_step=${MSTEP} \
        --quant=${QUANT} \
        --long=True \
        --resume-from auto'
#     CNT=`aws s3 ls s3://mv_pixart_ckpt/trained_model_zpp_512_e30_190x2kx${QUANT}_570k_n_stren_lr=${LR}_32x32_lam=${LAM}_all_step_lim=${MSTEP}_syth_lim=${SSTEP}_tri_long/ | wc -l`
#     if [ $CNT -ge 8 ] ; then
#         echo The command execute OK!
#         break;
#     fi
# done
