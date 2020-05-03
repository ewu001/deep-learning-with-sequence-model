# Neural machine translation in Pytorch

English to Spanish, trained on Nvidia GTX 750 Ti for 20 hours.

Encoder: bi-LSTM with reversed input sentence from training corpus.
Decoder: multiplicative attention, auto-regressive with beam search of 5.
Final BLEU score: 31.23.

Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository
