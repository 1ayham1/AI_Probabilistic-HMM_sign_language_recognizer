import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

from sklearn.model_selection import KFold


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):

        #with warnings.catch_warnings():
        #warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    Hints from the Forum:

        - Initializing the model to constant rather than null was also suggested in the forum by: katie_tiwari
        https://discussions.udacity.com/t/nonetype-object-has-no-attribute-n-components/247439/15

        - cross-validation with BIC and DIC selectors is not needed i.e. no need to split the training set as in SelectorCV.
            Evaluating on the same data is enough. Suggested by: angelmtenor & katie_tiwari
        https://discussions.udacity.com/t/split-into-train-and-test/230456
    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    REF:
    ----------------------------------------------------------------------------
    https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt
    ----------------------------------------------------------------------------
    L: Likelihood of the fitted model
    P: # of parameters

        p   = n*(n-1) + (n-1) + 2*d*n
            = n^2 + 2*d*n - 1

    where: d is the number of feautes & n is the number of n_components

    N: number of data points
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object

        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        largest_BIC = float("inf")   # return value of highest average C.V
        best_model= self.base_model(self.n_constant)         # the corrosponding model woth top_score


        for n_components in range(self.min_n_components, self.max_n_components + 1):

            try:

                #-------------------------------------------
                n = n_components
                d = len(self.X[0])  # number of features
                p =   (n **2) + (2*d*n) - 1 #n*(n-1) + (2*d*n)
                N = len(self.X)
                #-------------------------------------------

                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                #-------------------------------------------
                logN = np.log(N)

                current_BIC = -2*logL + p*logN

                if current_BIC < largest_BIC:
                    largest_BIC, best_model = current_BIC, model

            except:
                #print("Exception inside SelectorBIC")
                continue

        return best_model

#________________________________________________________________________________


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf


    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))


    NOTES: from the above ref.
    --------------------------
        - Penalize compleixities among other words. i.e. choose the model that is most likely to be distinctive
        - The anti-evidence is an evidence-like term computed by making the data set belongs to the competing class.
            - The anti-evidence measures the capacity of the corresponding model to generate data belonging to competing classes.
            - The ratio of the evidence and the anti-evidence is thus a measure of the model capacity to discriminate data from
              the two competing classes.
        - Discriminant Factor Criterion is the difference between the evidence of the model,
          given the corresponding data set, and the average over anti-evidences of the model.
        - By choosing the model which maximizes the evidence, and minimize the antievidences, the result is the best generative
          model for the correct class and the worst generative model for the competitive classes;
            this scheme thus selects the most discriminant models, resulting in an improved accuracy in regard to
            the classification task.

        - self.hwords contains a list for all words, and self.this_word stores the passed word of interest


    Ref:
    -----

    Comments on calculating the score: Thanks to Mohan-27
    https://discussions.udacity.com/t/dic-score-calculation/238907

    '''

    #--------------------------------------------------------------------
    def removekey(self, d, key):
        '''
            Removes one item from a dictionary and keeping the original one
            https://stackoverflow.com/questions/5844672/delete-an-item-from-a-dictionary
        '''

        r = dict(d)
        del r[key]
        return r

    #--------------------------------------------------------------------

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        largest_DIC = float('-inf')    # return value of highest average C.V
        best_model= self.base_model(self.n_constant)    # the corrosponding model woth top_score


        for n_components in range(self.min_n_components, self.max_n_components+1):


            anti_evidence = []       # Stores the Log Likelihood for all words but not the current one

            # hwords is a dictionary the key is the word, and the value is a tuple of (X, Lengths)
            # hword = {'word': (X, Lengths)}
            all_words_but_current = self.removekey(self.hwords, self.this_word)

            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)

                '''
                    The log(P(X(j)); where j != i is
                    just the model score when evaluating the model on all words other than the word
                    for which we are training this particular model.
                    So calculate the score for every other word and store them in a list so that you
                    take their avg later.
                '''

                anti_evidence = [model.score(v[0],v[1]) for k, v in all_words_but_current.items()]

            except:
                continue


            current_DIC = logL - np.mean(anti_evidence)

            if  current_DIC > largest_DIC:
                largest_DIC, best_model = current_DIC, model


        return best_model


#________________________________________________________________________________


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    REFERENCE:
    ----------
    Thanks to various suggestions found on udacity forums. Special Thanks to @angelmtenor
    https://discussions.udacity.com/t/understanding-better-model-selection/232987/11
    The following was adpated from the previous source:

        For each model combination (number of hidden states = 2,3, ....)

           For each fold:

              - Fit the model on training data
              - Get the score (Log Likelihood) for test data (of the fold) based on model
                (score of training data of the fold is not needed).

            After all folds: Average over the Log Likelihoods obtained

        After all model combinations, get the maximum average Log Likelihood and return the model

    '''


    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        largest_CV_Value = float("-inf") # return value of highest average C.V
        best_model = self.base_model(self.n_constant)  # the corrosponding model woth top_score

        n_splits = 3

        for n_components in range(self.min_n_components, self.max_n_components + 1):

            # don't split if the sequence is ver short
            if(len(self.sequences) < n_splits):
                break

            split_method = KFold(n_splits=n_splits)
            CV_Values = []       # Stores the Log Likelihood for a given fold

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                try:
                    #concatenate sequences referenced in an index list and returns tuple of the new X,lengths
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test,  lengths_test  = combine_sequences(cv_test_idx, self.sequences)

                    model = self.base_model(n_components)
                    logL = model.score(X_test, lengths_test)
                    CV_Values.append(logL)

                except:
                    #print("Something wrong happened: -- Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
                    continue


            current_CV_Avg = np.mean(CV_Values)  # The higher the better

            if  current_CV_Avg > largest_CV_Value:
                largest_CV_Value, best_model = current_CV_Avg, model

        return best_model
