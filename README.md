# transformers-into-vaes

Code for _Finetuning Pretrained Transformers into Variational Autoencoders_ (our submission to NLP Insights Workshop 2021), to be uploaded before 8/5.

# Running experiments:

1. Install dependencies.
```bash
pip install -r requirements.txt
```

2. Run phase 1 (encoder only training):
```
./run_encoder_training snli
```

3. Run phase 2 (full training):
```bash
./run_training snli <path_to_checkpoint_from_phase_1>
```
