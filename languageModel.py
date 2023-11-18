# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:36:01 2023

@author: mbelic
"""
import random
import torch
from metaphone import doublemetaphone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)


class LanguageProcessingModel:
    def __init__(self, model_name="t5-small", tokenizer_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_tokens = 50

    def _encoder(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        embeddings = self.model.encoder(inputs["input_ids"]).last_hidden_state
        return embeddings

    def _decoder(self, embeddings):
        decoder_start_token_id = self.model.config.decoder_start_token_id
        batch_size = 1
        ones_a = (batch_size, 1)
        ones_kw = {"dtype": torch.long, "device": self.model.device}
        decoder_input_ids = torch.ones(*ones_a, **ones_kw) * decoder_start_token_id

        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        generated_sequence = []
        for _ in range(self.max_tokens):
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=embeddings,
            )
            logits = self.model.lm_head(decoder_outputs["last_hidden_state"])
            probs = torch.softmax(logits, dim=-1)
            predicted_token_id = torch.argmax(probs, dim=-1)[:, -1].unsqueeze(-1)
            generated_sequence.append(predicted_token_id)
            decoder_input_ids = torch.cat(
                [decoder_input_ids, predicted_token_id], dim=-1
            )
        generated_sequence = torch.cat(generated_sequence, dim=-1)
        decoded_output = self.tokenizer.batch_decode(
            generated_sequence, skip_special_tokens=True
        )
        return decoded_output[0]

    def hinder_meaning(self, embeddings):
        embeddings[:, 4:11, :] += 0.32
        return embeddings

    def hinder_utterance(self, tokens):
        stop_words = set(stopwords.words("english"))
        content_tokens = [t for t in tokens if t.lower() not in stop_words]
        brocas_utterances = []
        i = 0
        while i < len(content_tokens):
            words_per_utterance = random.randint(1, 3)
            utterance = content_tokens[i:i + words_per_utterance]
            brocas_utterances.append(utterance)
            i += words_per_utterance
        brocas_aphasia_tokens = []
        fillers = ["", " ah, ", " er, ", " um, "]
        for utterance in brocas_utterances:
            repetition_chance = random.random()
            if repetition_chance < 0.3:
                repeat_word_index = random.randint(0, len(utterance) - 1)
                utterance.insert(repeat_word_index, utterance[repeat_word_index])
            random_filler = random.choices(fillers, weights=[5, 1, 1, 1], k=1)
            brocas_aphasia_tokens.extend(utterance)
            brocas_aphasia_tokens.extend(random_filler)
            brocas_aphasia_tokens.append("...")
        messed_up_tokens = brocas_aphasia_tokens[:-1]
        # Remove the last "..." from the tokens
        pronounced_sentence = "".join(messed_up_tokens)
        return messed_up_tokens, pronounced_sentence

    def comprehension(self, input_text, aphasia=None):
        embeddings = self._encoder(input_text)
        if aphasia == "Wernicke":
            embeddings = self.hinder_meaning(embeddings)
        gen_sentence = self._decoder(embeddings)
        return embeddings, gen_sentence

    def articulation(self, sentence, aphasia=None):
        tokens = self.tokenizer.tokenize(
            sentence, return_tensors="pt", add_special_tokens=False
        )
        if aphasia == "Broca":
            tokens, sentence = self.hinder_utterance(tokens)
        phonetic_representation = [doublemetaphone(t) for t in tokens]
        return phonetic_representation, sentence

    def process(self, in_sentence, aphasia=None, verbose=True):
        # Get meaning (Ventral stream)
        embeddings, gen_sentence = language_model.comprehension(in_sentence, aphasia)
        # Get phonetic representation (Dorsal stream)
        phonetic_representation, pronounced_sentence = language_model.articulation(
            gen_sentence, aphasia
        )
        pathology = "HEALTHY"
        if aphasia == "Wernicke":
            pathology = "WERNICKE'S APHASIA"
        elif aphasia == "Broca":
            pathology = "BROCA'S APHASIA"
        if verbose:
            print(f"{pathology:20}: {pronounced_sentence}\n{phonetic_representation}")
        return embeddings, gen_sentence, phonetic_representation


if __name__ == "__main__":

    language_model = LanguageProcessingModel()
    in_sentence = "My flower pot was empty again, so I went to buy a begonia."
    print("Intended sentence: ", in_sentence)
    for aphasia in [None, "Broca", "Wernicke"]:
        language_model.process(in_sentence, aphasia)
        print("============================================================")
