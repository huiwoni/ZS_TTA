########################## zero-shot / training-free few-shot classification
data_root='/home/huiwon/TDA/data'
# testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat
testsets=I
# num_important_channel=250/400/0 ## not activated when num_important_channel=0
num_important_channel=0
lambda_ape=0.3/0.7 ## not activated when num_important_channel=0
lr=0.0001 ## not activated when num_important_channel=0
epoch=20 ## not activated when num_important_channel=0
#arch=RN50
arch=ViT-B/16
bs=1
selection_p=1
ctx_init=a_photo_of_a

# for nshot in 0 1 2 4 8 16
for nshot in 0
do
  CUDA_LAUNCH_BLOCKING=1 python ./dummy_dmn_main.py ${data_root} --test_sets ${testsets} --selection_p ${selection_p}  \
  -a ${arch} -b ${bs}  --ctx_init ${ctx_init}  --memory_size 50 --text_prompt desciption_prompt  --log camera_ready_dmn_tf_description_searched_vit \
  --gpu 5 --n_shot ${nshot} --n_augview 0   --beta 5.5   --use_searched_param  \
  --num_important_channel ${num_important_channel} --lambda_ape ${lambda_ape} --epoch ${epoch} --lr ${lr}
done