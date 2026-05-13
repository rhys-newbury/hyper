set -eux

TRAIN_NPZ=data/ffhq_latents_6class/train_latents_by_class.npz
TEST_NPZ=data/ffhq_latents_6class/test_latents_by_class.npz

for EPS in 0.1 1.0 10.0; do
  TAG=$(echo "$EPS" | tr '.' 'p')

  python -m ffhq.reinforce_ffhq \
    --train-npz $TRAIN_NPZ \
    --test-npz  $TEST_NPZ \
    --run-root  runs/ffhq/eps_${TAG}_reinforce \
    --run-name  drift_ffhq \
    --d-z 512 --d-e 64 --hidden 1024 --n-hidden 3 \
    --steps 10000 \
    --nneg 2048 --npos 2048 \
    --lr 3e-4 --emb-lr 1e-3 \
    --temp 10.0 --maxent-coef 0.0 \
    --emd-every 50 --emd-samples 512 --log-every 50 \
    --save-every 500 \
    --seed 42

  python -m ffhq.drift_ffhq \
    --train-npz $TRAIN_NPZ \
    --test-npz $TEST_NPZ \
    --save-path runs/ffhq/eps_${TAG}_baseline/drift_ffhq_model.pt \
    --emd-plot runs/ffhq/eps_${TAG}_baseline/drift_ffhq_emd.png \
    --emd-perclass-plot runs/ffhq/eps_${TAG}_baseline/drift_ffhq_emd_perclass.png \
    --pca-plot runs/ffhq/eps_${TAG}_baseline/drift_ffhq_pca.png \
    --d-z 512 --d-e 64 --hidden 1024 --n-hidden 3 \
    --iters 1000 --batch-size 4096 --lr 3e-4 --emb-lr 1e-3 \
    --plan two-sided --eps $EPS --sinkhorn-iters 30 --dist l2_sq \
    --emd-every 50 --emd-samples 512 --log-every 50 \
    --seed 42

  python -m ffhq.drift_ffhq \
    --train-npz $TRAIN_NPZ \
    --test-npz $TEST_NPZ \
    --save-path runs/ffhq/eps_${TAG}_sinkhorn/drift_ffhq_model.pt \
    --emd-plot runs/ffhq/eps_${TAG}_sinkhorn/drift_ffhq_emd.png \
    --emd-perclass-plot runs/ffhq/eps_${TAG}_sinkhorn/drift_ffhq_emd_perclass.png \
    --pca-plot runs/ffhq/eps_${TAG}_sinkhorn/drift_ffhq_pca.png \
    --d-z 512 --d-e 64 --hidden 1024 --n-hidden 3 \
    --iters 1000 --batch-size 4096 --lr 3e-4 --emb-lr 1e-3 \
    --plan sinkhorn --eps $EPS --sinkhorn-iters 30 --dist l2_sq \
    --emd-every 50 --emd-samples 512 --log-every 50 \
    --seed 42
done
