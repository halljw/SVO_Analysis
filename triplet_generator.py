#!/usr/bin/env python3

import time
from nltk.tokenize import sent_tokenize
from stanford_parser_triplet_extractor import *

"""
Triplet Generator reads a text file, tokenizes it, and outputs the document
  as a collection of SVO triplets representative of the original text.
"""
class Triplet_Generator:

    def __init__(self):
        self.spte = Stanford_Parser_Triplet_Extractor()
        self.start = time.time()

    def clean_text(self, text):
        """
        Returns input text with newline characters removed, all text lowercase
        """
        return " ".join(text.replace('\n', ' ').lower().strip().split())

    def get_pos(self, pos_label):
        """
        Returns the appropriate POS label compatible with wordnet word senses.
        """
        if pos_label in ['NN', 'NNS', 'NNP', 'NNPS']:
            return 'n'
        if pos_label in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return 'v'
        return None

    def generate_sentence_triplet(self, text):
        """
        From input text, generates a list of tuplets.
        Each tuplet comprises a Sentence-Triplet pair.
        The sentence is the raw string from the text.
        The Triplet is a list of triplets extracted
        from that sentence
        >>> tg = Triplet_Generator()
        >>> tg.generate_sentence_triplet('A flower, sometimes known as a bloom or blossom, is the reproductive structure found in plants that are floral (plants of the division Magnoliophyta, also called angiosperms). The biological function of a flower is to effect reproduction, usually by providing a mechanism for the union of sperm with eggs.')
        [('a flower, sometimes known as a bloom or blossom, is the reproductive structure found in plants that are floral (plants of the division magnoliophyta, also called angiosperms).', [(('plants', 'NNS'), ('are', 'VBP'), ('floral', 'JJ'), False)]), ('the biological function of a flower is to effect reproduction, usually by providing a mechanism for the union of sperm with eggs.', [(('function', 'NN'), ('is', 'VBZ'), ('reproduction', 'NN'), False)])]
        >>> tg.generate_sentence_triplet('')
        []
        """
        return [(sent,self.spte.extract_triplets(self.spte.parse(sent))) for sent in sent_tokenize(self.clean_text(text))]

    def document_to_triplets(self, doc_name):
        """
        Takes the name of a document as input.
        Generates a document containing word sense triplets 
        extracted from the text of the document.
        """
        with open(doc_name, 'r') as f:
            self.text = f.read()
            print('Extracting triplets for ' + str(doc_name) + ':', time.time() - self.start)
        self.trips = self.generate_word_sense_triplets(self.generate_sentence_triplet(self.text))
        print('Triplets extracted:', time.time() - self.start)
        self.output_name = doc_name.replace('.txt','.triplets.txt')
        self.f = open(self.output_name, 'w')
        for trip in self.trips:
            self.f.write(str(trip[0]) + ',' + str(trip[1]) + ',' + str(trip[2]) + '\n')
        self.f.close()

if __name__=="__main__":
    import doctest
    doctest.testmod()
