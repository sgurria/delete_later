# README: Flirtify

## Overview

Flirtify is an NLP project that aims to classify text as flirty or neutral and then paraphrase the input text to make it more flirtatious or playful. 
This project is designed to help individuals who struggle with flirting by providing them with a tool that can help them craft more engaging and playful messages.

## Libraries

To run this project, we installed a packages such as numpy, pandas, nltk, sklearn, tensorflow, keras, evaluate, and the transformers from Hugging Face.

## Data

The data used in this project is a combination of publicly available datasets from tinder and our own generated content; it was difficult to find the 
data as it often came with warning of inappropriateness. 
For the purpose of this project, we created two datasets:

1. **ieuniversity/flirty_or_not:** The dataset contains two columns, one with the labels (flirty or neutral) and another one with the text messages. 
It has 2.11k rows and the it is balanced. This dataset was used to train the classification model.

2. **ieuniversity/neutral_to_flirty:** The dataset contains two columns, one with the text in a normal tone and another one with a flirty tone. 
It has 146 rows. This dataset was used to train the paraphrasing model for our specific task.

Moreover, we used an additional dataset **humarin/chatgpt-paraphrases** 20k rows of data for the general paraphraser model.

## Models

We built three models for this project:

1. **ieuniversity/flirty_classifier:** After trying with different pretrained models such as facebook/bart-large and Roberta, we obtained the best scores with DeBERTa-v3.
So, we used the DeBERTa-v3 transformer model and trained it with **ieuniversity/flirty_or_not** dataset. 
We did fine tuning and trained with 10 epochs. We had to add the parameters of fp16 and gradient_accumulation_steps to be able to run 
the model without crashing the Colab session. We obtained an F1-score of 0.70 on the train and 0.75 on the test sets. 
Furthermore, we pushed the model to Hugging Face to use it for later deployment.  

2. **ieuniversity/general_paraphrase:** After doing some reasearch and trying with different pretrained models such as T5-small, T5-base, ramsrigouthamg/t5_sentence_paraphraser 
and prithivida/parrot_paraphraser_on_T5, we obtained the best scores with humarin/chatgpt_paraphraser_on_T5_base. 
So, we used humarin/chatgpt_paraphraser_on_T5_base transformer model and trained it with 18k rows from 
the **humarin/chatgpt-paraphrases** dataset. We did fine tuning and trained with 5 epochs. 
We used a batch size for training and evaluation of 4, using 64 gradient_accumulation_steps. 
We obtained a Rouge-1 score of 64.64. Furthermore, we pushed the model to Hugging Face to use it for fine tuning our specific use case. 

3. **ieuniversity/flirty_paraphraser:** After retrieving our dataset **ieuniversity/neutral_to_flirty**, 
we fine tuned the **ieuniversity/general_paraphrase** model so that it could be implemented to 
flirtify as a way of data augmentation. We did fine tuning and trained with 10 epochs. 
We used a batch size for training and evaluation of 16, using 64 gradient_accumulation_steps. 
We obtained a Rouge-1 score of 19.88. Furthermore, we pushed the model to Hugging Face to use it for later deployment.

## Architecture 

![WhatsApp Image 2023-04-11 at 21 41 23](https://user-images.githubusercontent.com/94801284/231271335-40d0c23e-0fc3-4978-b2a4-96c8f44151cd.jpeg)

## Gradio app.py

We made the model checkpoints with ieuniversity/flirty_classifier and ieuniversity/general_paraphrase. 
Then, we defined a function to create our desired output and designed our Gradio interface. 
This was uploaded to our HuggingFace Space **ieuniversity/flirtify** as an app.py to deploy the model.

## Conclusion  

- Flirtify is a successful NLP project that provides a tool to help individuals who struggle with flirting to 
craft more engaging and playful messages.
- The project uses two datasets, including a combination of publicly available datasets from 
Tinder and their own generated content, to train the classification and paraphrasing models.
- Three models were built and fine-tuned using the datasets: ieuniversity/flirty_classifier, 
ieuniversity/general_paraphrase, and ieuniversity/flirty_paraphraser.
- The Flirtify project has been deployed using Gradio and Hugging Face.

# Recommendations
- As the flirty_paraphraser model achieved a lower Rouge-1 score than the general_paraphrase model, 
it might be worth exploring other models or fine-tuning the current one with a larger dataset to improve its performance.
- It is recommended to keep updating the datasets used for training and fine-tuning the models to improve 
their accuracy and performance.
- Since the project is designed to help people with flirting, it might be useful to provide some guidance or 
recommendations on how to use the tool effectively, especially for people who are new to flirting or online dating. 
This could be done by creating a user guide or adding some tips and tricks to the Gradio interface.

## Credits

This project was developed by Cyril Tabet, Miquel Amengual and Sophia Gurria Hamdan.

