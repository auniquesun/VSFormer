# optimize VSFormer on RGBD with 
# 1. coswarm lr strategy
# 2. 12 views
# 3. 4 attn. blocks, 8 heads, model dimension 512 
# 4. 2-layer `cls_head`
# 5. `stage_one` and `stage_two` 
# 6. 10-fold cross-validation

PROJ_NAME=VSFormer
MAIN_PROGRAM=main_rgbd.py
MODEL_NAME=vsformer.py
BASE_MODEL_NAME=alexnet
BASE_FEATURE_DIM=512

TASK=CLS
DATASET=RGBD
TRAIN_PATH=data/rgbd-dataset_eval
TEST_PATH=data/rgbd-dataset_eval
NUM_CLASSES=51
NUM_VIEWS=12
BATCH_SIZE=750
BASE_MODEL_BATCH_SIZE=4500
PRINT_FREQ=10
NUM_WORKERS=3

NUM_LAYERS=4
NUM_HEADS=8
NUM_CHANNELS=512
MLP_RATIO=2
MAX_DPR=0.0
ATTEN_DROP=0.1
MLP_DROP=0.5
CLSHEAD_LAYERS=2

# 1st trial
EXP_NAME=RGBD-V12-L4H8D512-MR2-Alex-1_1
pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
    --proj_name ${PROJ_NAME} \
    --exp_name ${EXP_NAME} \
    --main_program ${MAIN_PROGRAM} \
    --model_name ${MODEL_NAME} \
    --shell_name scripts/${TASK}/${PROJ_NAME}/RGBD-V12-L4H8D512-MR2-Alex-1.sh \
    --dataset ${DATASET} \
    --num_obj_classes ${NUM_CLASSES} \
    --task ${TASK} \
    --stage_one --stage_two \
    --train_path ${TRAIN_PATH} \
    --test_path ${TEST_PATH} \
    --trial_id 1 \
    --base_model_name ${BASE_MODEL_NAME} \
    --resume --base_model_weights runs/${TASK}/${PROJ_NAME}/${EXP_NAME}/weights/sv_model_best.pth \
    --num_views ${NUM_VIEWS} \
    --epochs 300 \
    --base_model_epochs 30 \
    --batch_size ${BATCH_SIZE} \
    --base_model_batch_size ${BASE_MODEL_BATCH_SIZE} \
    --test_batch_size ${BATCH_SIZE} \
    --base_model_test_batch_size ${BASE_MODEL_BATCH_SIZE} \
    --print_freq ${PRINT_FREQ} \
    --optim adamw \
    --base_model_optim sgd \
    --lr 0.001 \
    --base_model_lr 0.01 \
    --lr_scheduler coswarm \
    --step_size 100 \
    --max_lr 0.001 --min_lr 0.0 --warm_epochs 5 --gamma 0.6 \
    --base_model_lr_scheduler cos \
    --num_workers ${NUM_WORKERS} \
    --num_layers ${NUM_LAYERS} --num_heads ${NUM_HEADS} --num_channels ${NUM_CHANNELS} \
    --mlp_widen_factor ${MLP_RATIO} --max_dpr ${MAX_DPR} --atten_drop ${ATTEN_DROP} --mlp_drop ${MLP_DROP} \
    --clshead_layers ${CLSHEAD_LAYERS}

# trial 2-10, use pretrained alexnet weights in 1st trial 
for i in 2 3 4 5 6 7 8 9 10
do
    EXP_NAME=RGBD-V12-L4H8D512-MR2-Alex-1_$i
    pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
        --proj_name ${PROJ_NAME} \
        --exp_name ${EXP_NAME} \
        --main_program ${MAIN_PROGRAM} \
        --model_name ${MODEL_NAME} \
        --shell_name scripts/${TASK}/${PROJ_NAME}/RGBD-V12-L4H8D512-MR2-Alex-1.sh \
        --dataset ${DATASET} \
        --num_obj_classes ${NUM_CLASSES} \
        --task ${TASK} \
        --stage_two \
        --train_path ${TRAIN_PATH} \
        --test_path ${TEST_PATH} \
        --trial_id $i \
        --base_model_name ${BASE_MODEL_NAME} \
        --resume --base_model_weights runs/${TASK}/${PROJ_NAME}/RGBD-V12-L4H8D512-MR2-Alex-1_1/weights/sv_model_best.pth \
        --num_views ${NUM_VIEWS} \
        --epochs 300 \
        --base_model_epochs 30 \
        --batch_size ${BATCH_SIZE} \
        --base_model_batch_size ${BASE_MODEL_BATCH_SIZE} \
        --test_batch_size ${BATCH_SIZE} \
        --base_model_test_batch_size ${BASE_MODEL_BATCH_SIZE} \
        --print_freq ${PRINT_FREQ} \
        --optim adamw \
        --base_model_optim sgd \
        --lr 0.001 \
        --base_model_lr 0.01 \
        --lr_scheduler coswarm \
        --step_size 100 \
        --max_lr 0.001 --min_lr 0.0 --warm_epochs 5 --gamma 0.6 \
        --base_model_lr_scheduler cos \
        --num_workers ${NUM_WORKERS} \
        --num_layers ${NUM_LAYERS} --num_heads ${NUM_HEADS} --num_channels ${NUM_CHANNELS} \
        --mlp_widen_factor ${MLP_RATIO} --max_dpr ${MAX_DPR} --atten_drop ${ATTEN_DROP} --mlp_drop ${MLP_DROP} \
        --clshead_layers ${CLSHEAD_LAYERS}
done