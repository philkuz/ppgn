#/bin/bash
# This file recreates an abstract picture of a bellpepper using ppgn
# and randomly sampling points in the graph.
# Phillip Kuznetsov <philkuz@ml.berkeley.edu>
# 2018

opt_layer=fc6
act_layer=fc8
units="945"      # Index of neurons in fc layers or channels in conv layers
xy=0              # Spatial position for conv layers, for fc layers: xy = 0

n_iters=200     # Run for N iterations
reset_every=0     # Reset the code every N iterations (for diversity)
save_every=5      # Save a sample every N iterations
lr=1
lr_end=1          # Linearly decay toward this ending lr (e.g. for decaying toward 0, set lr_end = 1e-10)
threshold=0       # Filter out samples below this threshold e.g. 0.98

# -----------------------------------------------
# Multipliers in the update rule Eq.11 in the paper
# -----------------------------------------------
epsilon1=1e-5        # prior
epsilon2=0           # condition
epsilon3=1e-10       # noise

# -----------------------------------------------
# Ablative Compression variables
edge_epsilon=0   # edge
content_epsilon=0 #1 # content
style_epsilon=0 #1 # content
mask_epsilon=1  # mask epsilon
mask_type='laplace'
ratio_sample=0.03 # amount to sample if using random mask sampling
content_layer=conv4 # layer to use for content loss6
init_dir="images/"    # Start from a random code. To start from a real code, replace with a path e.g. "images/dirname.jpg"
# Condition net
net_weights="nets/caffenet/bvlc_reference_caffenet.caffemodel"
net_definition="nets/caffenet/caffenet.prototxt"
#-----------------------

# Output dir
output_dir="output/masks_black/"
mkdir -p ${output_dir}

# Directory to store samples
if [ "${save_every}" -gt "0" ]; then
    sample_dir=${output_dir}/samples
    rm -rf ${sample_dir}
    mkdir -p ${sample_dir}
fi

for unit in ${units}; do
    unit_pad=`printf "%04d" ${unit}`
    seed=0
    #for seed in {0..0}; do

    python ./sampling_masks.py \
        --act_layer ${act_layer} \
        --opt_layer ${opt_layer} \
        --unit ${unit} \
        --xy ${xy} \
        --n_iters ${n_iters} \
        --save_every ${save_every} \
        --reset_every ${reset_every} \
        --lr ${lr} \
        --lr_end ${lr_end} \
        --seed ${seed} \
        --output_dir ${output_dir} \
        --init_dir ${init_dir} \
        --mask_type ${mask_type} \
        --ratio_sample ${ratio_sample} \
        --epsilon1 ${epsilon1} \
        --epsilon2 ${epsilon2} \
        --epsilon3 ${epsilon3} \
        --mask_epsilon ${mask_epsilon} \
        --edge_epsilon ${edge_epsilon} \
        --content_epsilon ${content_epsilon} \
        --style_epsilon ${style_epsilon} \
        --content_layer ${content_layer} \
        --threshold ${threshold} \
        --net_weights ${net_weights} \
        --net_definition ${net_definition} \
        --use_square

    #done
done
