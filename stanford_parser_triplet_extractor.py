#!/usr/bin/env python3

from nltk.parse.stanford import StanfordDependencyParser
import os
os.environ['JAVAHOME'] = '/usr/lib/jvm/java-7-openjdk-amd64/bin'

"""
The Stanford Parser Triplet Extractor class uses the Stanford Dependency Parser to extract
  the subject-verb-object triplets present in a given sentence. Results are returned as a list
  of results, where each result contains four elements:

  (Subj, POS), (Verb, POS), (Obj, POS), Negation

SVO triplets are taken to include action triplets with transitive verbs as the verb head and
  copula constructions (generally with "to be" as the head verb). Sentences containing
  conjunctions are parsed into multiple triplet results, treating the sentence as if it
  contained multiple allowable SVO triplets instead of a conjuction. If any negation word is
  present in the sentence, the SVO triplet is marked as negated (negation = TRUE).
"""
class Stanford_Parser_Triplet_Extractor:

    def __init__(self):
        self.path_to_jar='StanfordParser/stanford-parser-full-2015-04-20/stanford-parser.jar'
        self.path_to_models_jar='StanfordParser/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'
        self.dependency_parser=StanfordDependencyParser(path_to_jar=self.path_to_jar, path_to_models_jar=self.path_to_models_jar)

        self.subj_targets = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'xsubj']
        self.verb_targets = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.noun_targets = ['NN', 'NNS', 'NNP']
        self.obj_targets = ['dobj']
        self.wh_subj = ['WDT', 'WP', 'WP$', 'WRB']

    def parse(self,text):
        if not text:
            return []
        return list(next(self.dependency_parser.raw_parse(text)).triples())

    def extract_triplets(self, parsed_text):
        """
        Use the Stanford Dependency Parser to parse text
        Return a list of SVO triplets and associated negation
        >>> spte = Stanford_Parser_Triplet_Extractor()
        >>> spte.extract_triplets(spte.parse(''))
        []

        >>> spte.extract_triplets(spte.parse('Bill runs quickly.'))
        []

        >>> spte.extract_triplets(spte.parse('Bill runs and jumps.'))
        []

        >>> spte.extract_triplets(spte.parse("Bill does not eat fish."))
        [(('Bill', 'NNP'), ('eat', 'VB'), ('fish', 'NN'), True)]

        >>> spte.extract_triplets(spte.parse('Bill and Tom kick ass.'))
        [(('Tom', 'NNP'), ('kick', 'VBP'), ('ass', 'NN'), False), (('Bill', 'NNP'), ('kick', 'VBP'), ('ass', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Mary went to play tennis.'))
        [(('Mary', 'NNP'), ('went', 'VBD'), ('play', 'VB'), False), (('Mary', 'NNP'), ('play', 'VB'), ('tennis', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Bill kicks ass and takes names.'))
        [(('Bill', 'NNP'), ('kicks', 'VBZ'), ('ass', 'NN'), False), (('Bill', 'NNP'), ('takes', 'VBZ'), ('names', 'NNS'), False)]

        >>> spte.extract_triplets(spte.parse('Bill is big and fat.'))
        [(('Bill', 'NNP'), ('is', 'VBZ'), ('big', 'JJ'), False), (('Bill', 'NNP'), ('is', 'VBZ'), ('fat', 'JJ'), False)]

        >>> spte.extract_triplets(spte.parse('Bill was big and fat.'))
        [(('Bill', 'NNP'), ('was', 'VBD'), ('big', 'JJ'), False), (('Bill', 'NNP'), ('was', 'VBD'), ('fat', 'JJ'), False)]

        >>> spte.extract_triplets(spte.parse('Bill is big and kicks ass.'))
        [(('Bill', 'NNP'), ('is', 'VBZ'), ('big', 'JJ'), False), (('Bill', 'NNP'), ('kicks', 'VBZ'), ('ass', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Bill and Tom take names and chew bubblegum.'))
        [(('Tom', 'NNP'), ('take', 'VBP'), ('names', 'NNS'), False), (('Tom', 'NNP'), ('chew', 'VBP'), ('bubblegum', 'NN'), False), (('Bill', 'NNP'), ('take', 'VBP'), ('names', 'NNS'), False), (('Bill', 'NNP'), ('chew', 'VBP'), ('bubblegum', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Bill and Tom are fat and eat cheese.'))
        [(('Tom', 'NNP'), ('are', 'VBP'), ('fat', 'JJ'), False), (('Tom', 'NNP'), ('eat', 'VBP'), ('cheese', 'NN'), False), (('Bill', 'NNP'), ('are', 'VBP'), ('fat', 'JJ'), False), (('Bill', 'NNP'), ('eat', 'VBP'), ('cheese', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Bill and Tom eat cheese and are fat.'))
        [(('Tom', 'NNP'), ('eat', 'VBP'), ('cheese', 'NN'), False), (('Tom', 'NNP'), ('are', 'VBP'), ('fat', 'JJ'), False), (('Bill', 'NNP'), ('eat', 'VBP'), ('cheese', 'NN'), False), (('Bill', 'NNP'), ('are', 'VBP'), ('fat', 'JJ'), False)]

        >>> spte.extract_triplets(spte.parse('Tom eats fish and cheese.'))
        [(('Tom', 'NNP'), ('eats', 'VBZ'), ('fish', 'NN'), False), (('Tom', 'NNP'), ('eats', 'VBZ'), ('cheese', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Tom likes to eat fish.'))
        [(('Tom', 'NNP'), ('likes', 'VBZ'), ('eat', 'VB'), False), (('Tom', 'NNP'), ('eat', 'VB'), ('fish', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Bill is an honest man.'))
        [(('Bill', 'NNP'), ('is', 'VBZ'), ('man', 'NN'), False)]

        >>> spte.extract_triplets(spte.parse('Dole was defeated by Clinton.'))
        [(('Dole', 'NNP'), ('was', 'VBD'), ('defeated', 'VBN'), False), (('Clinton', 'NNP'), ('defeated', 'VBN'), ('Dole', 'NNP'), False)]

        >>> spte.extract_triplets(spte.parse('She looks very beautiful.'))
        [(('She', 'PRP'), ('looks', 'VBZ'), ('beautiful', 'JJ'), False)]

        >>> spte.extract_triplets(spte.parse('He says that you like to swim.'))
        [(('you', 'PRP'), ('like', 'VBP'), ('swim', 'VB'), False)]

        >>> spte.extract_triplets(spte.parse("Last night, I shot an elephant in my pajamas. What the elephant was doing in my pajamas I'll never know."))
        [(('I', 'PRP'), ('shot', 'VBD'), ('elephant', 'NN'), False), (('elephant', 'NN'), ('doing', 'VBG'), ('pajamas', 'NNS'), False)]

        >>> spte.extract_triplets(spte.parse('A rare black squirrel has become a regular visitor to a suburban garden.'))
        [(('squirrel', 'NN'), ('become', 'VBN'), ('visitor', 'NN'), False)]
        """

        triplets = []
        for index,item in enumerate(parsed_text):
            neg = False
            if item[1] == 'acl:relcl':
                rel_clause_subj = item[0]

            elif item[1] in self.subj_targets:

                if item[2][1] in self.wh_subj:
                    try:
                        subj = rel_clause_subj
                    except:
                        subj = item[2]
                else:
                    subj = item[2]

                verb = item[0]

                for index_1, obj in enumerate(parsed_text[index:]):

                    if obj[1] == 'neg':
                        neg = True

                    elif obj[1] in self.obj_targets and obj[0] == verb and verb[1] in self.verb_targets:
                        triplets.append((subj,verb,obj[2],neg))

                    elif obj[1] == 'nmod' and obj[0] == verb and verb[1] in self.verb_targets:
                        skip = False
                        for triplet in triplets:
                            if triplet[0] == subj or triplet[1] == verb:
                                skip = True
                        if not skip:
                            triplets.append((subj,verb,obj[2],neg))

                    elif obj[1] == 'auxpass' and obj[0] == verb:
                        triplets.append((subj,obj[2],verb,neg))
                        for obj_1 in parsed_text[index:][index_1:]:
                           if obj_1[0] == obj[0] and obj_1[1] == 'nmod':
                                triplets.append((obj_1[2],obj_1[0],subj,neg))

                    elif obj[1] == 'xcomp' and obj[0] == verb and verb[1] in self.verb_targets:
                        triplets.append((subj,verb,obj[2],neg))
                        xcomp_triplet = self.extract_xcomp_triplet(parsed_text[index:],subj,obj[2],neg)
                        if xcomp_triplet:
                            triplets.append(xcomp_triplet)

                    elif obj[1] == 'cop' and obj[0] == verb:
                        triplets.append((subj,obj[2],verb,neg))

                    elif obj[1] == 'conj':
                        if obj[2][1] in self.noun_targets and obj[0] == item[2]:
                            skip=parsed_text.index(obj)
                            triplets += self.extract_triplets([(item[0], 'nsubj', obj[2])] + parsed_text[index+index_1:skip] + parsed_text[skip+1:])

                        elif obj[0][1] in self.noun_targets and obj[2][1] in self.noun_targets:
                            verb = self.find_conj_verb(parsed_text, index+index_1, obj[0])
                            if verb:
                                triplets += self.extract_triplets(parsed_text[:index_1] + [(verb, 'dobj', obj[2])])
                            if obj[0][1] in self.verb_targets and obj[2][1] in self.verb_targets:
                                skip=parsed_text.index(obj)
                                triplets += self.extract_triplets([(obj[2], 'nsubj', item[2])] + parsed_text[index+index_1:skip] + parsed_text[skip+1:])

                        elif obj[2][1] in self.verb_targets and obj[0] == item[0]:
                            skip=parsed_text.index(obj)
                            triplets += self.extract_triplets([(obj[2], 'nsubj', item[2])] + parsed_text[index+index_1:skip] + parsed_text[skip+1:])

                        elif obj[0][1] == 'JJ' and obj[2][1] == 'JJ':
                            copula = self.find_conj_copula(parsed_text,index_1,obj[0])
                            triplets.append((subj,copula,obj[2],neg))

                        elif obj[2][1] == 'JJ' and obj[0] == item[0]:
                            triplets += self.extract_triplets([(obj[2], 'nsubj', item[2])] + parsed_text[index_1:])

        self.output = []
        [self.output.append(triplet) for triplet in triplets if triplet not in self.output]
        return self.output


    def find_conj_copula(self, results, index, target):
        for item in [results[i] for i in range(index-1,-1,-1)]:
            if item[1] == 'cop' and item[0] == target:
                return item[2]

    def find_conj_verb(self, results, index, target):
        for item in [results[i] for i in range(index-1,-1,-1)]:
            if item[1] in self.obj_targets and item[2] == target:
                return item[0]

    def extract_xcomp_triplet(self, results, subj, verb,neg):
        for item in results:
            if item[1] in self.obj_targets and item[0] == verb:
                return (subj,verb,item[2],neg)

if __name__=="__main__":
    import doctest
    doctest.testmod()

    txt='An optical writing apparatus comprising: a receiving unit configured to receive an input of a second image data which is formed by superposing an unauthorized copy protection pattern on a first image data; an unauthorized copy protection pattern recognition unit configured to recognize the unauthorized copy protection pattern in the second image data; a control unit configured to correct image data of the unauthorized copy protection pattern in pixel unit, and control a size of an isolated dot included in the unauthorized copy protection pattern; and a writing unit configured to write a corresponding image on a photosensitive body based on the thus-corrected image data, wherein the control unit is configured to correct data in the second image data having the unauthorized copy protection pattern recognized by the unauthorized copy protection pattern recognition unit based on predetermined data according to a standard of the unauthorized copy protection pattern and at least characteristics of the writing unit'
    spte = Stanford_Parser_Triplet_Extractor()
    print(spte.extract_triplets(spte.parse(txt)))
