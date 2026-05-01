AE_CKPT=runs/mnist_ae/20260501_164900_latent6/ae_final.pt

COMMON="--ae-ckpt $AE_CKPT \
        --nneg 64 --npos 64 --nuncond 16 --steps 5000 \
        --lr 0.0002 --weight-decay 0.01 \
        --warmup-steps 750 --grad-clip 2.0 --ema-decay 0.999 \
        --omega-min 1.0 --omega-max 4.0 --omega-exponent 3.0 \
        --dist-metric geodesic_sq --seed 0"
# Sinkhorn
for tag_rho in "tau0p0.3 0.3" "tau0p0.5 0.5" "tau0p1.0 1.0"; do
    tag=$(echo $tag_rho | cut -d' ' -f1)
    rho=$(echo $tag_rho | cut -d' ' -f2)
    python -m mnist.train_drifting $COMMON \
        --coupling sinkhorn --drift-form split \
        --sinkhorn-iters 20 --sinkhorn-marginal weighted_cols \
        --temps $rho --run-name mnist_sinkhorn_${tag}_l2sq
done
