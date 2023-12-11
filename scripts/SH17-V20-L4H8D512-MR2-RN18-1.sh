# optimize VSFormer on SHREC17 with 
# 1. coswarm lr strategy
# 2. 20 views
# 3. 4 attn. blocks, 8 heads, model dimension 512 
# 4. 2-layer `cls_head`
# 5. `stage_one` and `stage_two` 

PROJ_NAME=VSFormer
EXP_NAME=SH17-V20-L4H8D512-MR2-RN18-1
MAIN_PROGRAM=main_shrec17.py
MODEL_NAME=vsformer.py
BASE_MODEL_NAME=resnet18
BASE_FEATURE_DIM=512

TASK=RET
TRAIN_LABEL=train.csv
TEST_LABEL=val.csv
VERSION=normal

DATASET=SHREC17
NUM_CLASSES=55
NUM_VIEWS=20
BATCH_SIZE=204
BASE_MODEL_BATCH_SIZE=2400
PRINT_FREQ=100

NUM_WORKERS=2
NUM_LAYERS=4
NUM_HEADS=8
NUM_CHANNELS=512
MLP_RATIO=2
MAX_DPR=0.0
ATTEN_DROP=0.1
MLP_DROP=0.5
CLSHEAD_LAYERS=2

pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
    --proj_name ${PROJ_NAME} \
    --exp_name ${EXP_NAME} \
    --main_program ${MAIN_PROGRAM} \
    --model_name ${MODEL_NAME} \
    --shell_name scripts/${TASK}/${PROJ_NAME}/${EXP_NAME}.sh \
    --dataset ${DATASET} \
    --num_obj_classes ${NUM_CLASSES} \
    --task ${TASK} --train_label ${TRAIN_LABEL} --test_label ${TEST_LABEL} --shrec_version ${VERSION} \
    --stage_one --freeze_base_model --stage_two \
    --base_model_name ${BASE_MODEL_NAME} \
    --base_feature_dim ${BASE_FEATURE_DIM} \
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
    --lr_scheduler cos \
    --base_model_lr_scheduler cos \
    --num_workers ${NUM_WORKERS} \
    --num_layers ${NUM_LAYERS} --num_heads ${NUM_HEADS} \
    --num_channels ${NUM_CHANNELS} \
    --mlp_widen_factor ${MLP_RATIO} \
    --max_dpr ${MAX_DPR} --atten_drop ${ATTEN_DROP} --mlp_drop ${MLP_DROP} \
    --clshead_layers ${CLSHEAD_LAYERS} 
    