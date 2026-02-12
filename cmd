python -m demucs     --raw nobobby     --tasnet     --epochs 1    --batch_size 1     --samples 220500     --audio_channels 2     --logs logs     --checkpoints checkpoints     --models models



python -m demucs \
  --tasnet \
  --wav mydata \
  --no_augment \
  --repitch 0 \
  --shifts 0 \
  --overlap 0 \
  -w 0 \
  -b 1 \
  -e 2



python -m demucs \
  --tasnet \
  --wav mydata \
  --audio_channels 2 \
  --epochs 100 \
  --batch_size 1 \
  --samples 110250 \
  --repeat 100 \
  --lr 1e-3 \
  --no_augment




python -m demucs --tasnet --wav mydata --audio_channels 2 --epochs 2 --batch_size 1 --samples 110250 --repeat 100 --lr 1e-3 --no_augment
