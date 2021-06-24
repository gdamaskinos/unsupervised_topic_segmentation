# Unsupervised Topic Segmentation of Meetings with BERT Embeddings

This is the code for the paper **Unsupervised Topic Segmentation of Meetings with BERT Embeddings**.

The code doesn't require training and uses a pretrained model from https://huggingface.co/transformers/model_doc/roberta.html
See paper appendix for more information.

The entry point is `eval.eval_topic_segmentation` that returns the scores reported in the paper for the different methods.
