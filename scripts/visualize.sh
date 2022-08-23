model=$1
data_set=$2
output_dir=$3
use_gauss=$4
modeldir=$5
modelfile=$6

data_path=datasets
dim=192
batch_size=60

if [ "$model" = "deit_small_patch16_224" ]
then
    reserve_layer_idx=11
elif [ "$model" = "deit_tiny_patch16_224" ]
then
    reserve_layer_idx=11
elif [ "$model" = "cait_xxs24_224" ]
then
    reserve_layer_idx=1
fi

if [ "$data_set" = "CUB2011U" ]
then
    global_proto_per_class=10
    prototype_num=2000
    reserve_token_nums=81
elif [ "$data_set" = "Car" ]
then
    global_proto_per_class=5
    prototype_num=1960
    reserve_token_nums=121
elif [ "$data_set" = "Dogs" ]
then
    global_proto_per_class=5
    prototype_num=1200
    reserve_token_nums=81
fi

python main_visualize.py \
    --finetune=visualize \
    --modeldir=$modeldir \
    --model=$modelfile \
    --data_set=$data_set \
    --data_path=$data_path \
    --prototype_shape $prototype_num $dim 1 1 \
    --reserve_layers=$reserve_layer_idx \
    --reserve_token_nums=$reserve_token_nums \
    --use_global=True \
    --use_ppc_loss=True \
    --global_coe=0.5 \
    --global_proto_per_class=$global_proto_per_class \
    --base_architecture=$model \
    --batch_size=$batch_size \
    --visual_type=slim_gaussian \
    --output_dir=$output_dir \
    --use_gauss=$use_gauss \
    --vis_classes 6 8 10 12 14