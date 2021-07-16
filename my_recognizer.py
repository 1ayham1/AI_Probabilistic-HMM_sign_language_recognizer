import warnings
from asl_data import SinglesData
import pandas as pd
from functools import partial


def get_score(x, c_model):
    return c_model.score(x['Seq'],x['Length'])


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """


    '''
       GENERAL IDEAS:
       --------------
       https://discussions.udacity.com/t/recognizer-implementation/234793/3
        - For every test set, get the corrosponding X and Lengths
            - For every word in a model:
                - get  (LogL) by computing the score
                - Store the corrosponding probabilities in a dictionary per word
                - get the highest probability and its word
            - Build (probabilities & guesses ) in the way described above.

        Thanks for the usefule and constructive discussions in:
            https://discussions.udacity.com/t/recognizer-implementation/234793
            https://discussions.udacity.com/t/failure-in-recognizer-unit-tests/240082/5
    '''

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []


    result = pd.DataFrame()
    sequence_df = pd.DataFrame.from_dict(test_set.get_all_Xlengths(), orient='index').reset_index()
    sequence_df.columns = ['Index','Seq','Length']


    for word, current_model in models.items():

        try:

            count_word = partial(get_score, c_model = current_model)
            result[word] = sequence_df.apply(count_word, axis=1)

        except:
            result[word]  = float("-inf")


    #print(result.head())
    probabilities = result.to_dict('records') #LogL for every row
    guesses = result.idxmax(axis=1).tolist()


    return probabilities, guesses
