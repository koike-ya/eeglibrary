
python experiment.py --batch-size 64 --model-type cnn --cuda --n-jobs 1 --expt-id debug \
--epoch-rate 1.0 --epochs 20 --tensorboard --sample-balance same --test --n-mels 64 \
--return-prob --loss-func ce

for model in rnn cnn_rnn
do
  python experiment.py --window-size 0.08 --window-stride 0.06 --batch-size 64 --model-type $model \
  --cuda --n-jobs 1 --expt-id debug --epoch-rate 1.0 --epochs 20 --tensorboard --sample-balance same \
  --test --n-mels 64 --return-prob --loss-func ce
done
