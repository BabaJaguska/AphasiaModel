# AphasiaModel
Dual Stream Model of language processing. Conceptual Broca's and Wernicke's aphasia model within the dual stream framework.

![dual stream diagram](https://github.com/BabaJaguska/AphasiaModel/assets/10914239/7ce24305-ff2b-4e1b-8fd9-4d23b20f065d)

In general: 
The Auditory Cortex sends signals to Wernicke's Area.
Wernicke's Area performs semantic processing and sends information to the Angular Gyrus for semantic comprehension.
Wernicke's Area also performs phonological processing and sends information to the Supramarginal Gyrus for phonological comprehension.
Both the Angular Gyrus and Supramarginal Gyrus send their processed information to Broca's Area.
Broca's Area is responsible for speech production and sends signals to the Motor Cortex.
The Motor Cortex controls the muscles involved in speech, leading to the production of speech.

In this repo:
Comprehension is scrambled (ventral stream) by perturbing the embeddings of a T5 model
Speech production (dorsal stream) is represented by adding fillers to disrupt utterance and excluding stop words to imitate telegraphic speech.
Phonetics are captured using Metaphone's phonetic representation.
