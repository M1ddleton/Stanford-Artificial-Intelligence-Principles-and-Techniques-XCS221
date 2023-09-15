#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass
    # ### START CODE HERE ###
    a = Counter(x.split())
    return dict(a)
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
        trainExamples: List[Tuple[T, int]],
        validationExamples: List[Tuple[T, int]],
        featureExtractor: Callable[[T], FeatureVector],
        numEpochs: int,
        eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.
    You should implement stochastic gradient descent.
    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight

    # ### START CODE HERE ###
    def predict(x):
        phi = featureExtractor(x)
        if dotProduct(weights, phi) < 0.0:
            return -1
        else:
            return 1

    for i in range(numEpochs):
        for item in trainExamples:
            x, y = item
            phi = featureExtractor(x)
            temp = dotProduct(weights, phi) * y
            if temp < 1: increment(weights, -eta * -y, phi)
        print("Iteration:%s, Training error:%s, Test error:%s" % (
        i, evaluatePredictor(trainExamples, predict), evaluatePredictor(validationExamples, predict)))

    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        # ### START CODE HERE ###
        phi = {}
        for x in weights.keys():
            num = random.randint(-10, 10)
            if num > 0:
                phi[x] = num
        sum_of_weights = 0
        for weight in weights.keys():
            if weight in phi.keys():
                sum_of_weights += weights[weight]*phi[weight]

        y = 1 if sum_of_weights >= 0 else -1
        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        pass
        # ### START CODE HERE ###
        word = "".join(x.split()).lower()
        Dict = {}
        for i in range(len(x) + 1 - n):
            n_gram = word[i:i + n]
            if len(n_gram) == n:
                  Dict[n_gram] = Dict.get(n_gram, 0) + 1
        return Dict
        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from solution import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
                "Official: train error = %s, validation error = %s"
                % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
        examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """

    # ### START CODE HERE ###
    def distance(x1, x2):
        result = 0
        for f, v in x2.items():
            result += (x1.get(f, 0) - v) ** 2
        return result

    def average(points):
        n = len(points) #used to be float(len(points))
        result = {}
        for p in points:
            increment(result, 1 / n, p)
        return result

    centroids = random.sample(examples, K)
    old_assignments = []
    for _ in range(maxEpochs):
        centr_point_pair = []
        assignments = []
        totalCost = 0

        for p in examples:
            dist = [distance(c, p) for c in centroids]
            new_centr = dist.index(min(dist))
            assignments.append(new_centr)
            totalCost += min(dist)
            centr_point_pair.append((new_centr, p))
        if assignments == old_assignments:
            break
        else:
            old_assignments = list(assignments)
        centr_point_pair = sorted(centr_point_pair, key=lambda item: item[0])

        new_centroids = []
        import itertools
        for key, kpList in itertools.groupby(centr_point_pair, key=lambda item: item[0]):
            pList = [kp[1] for kp in kpList]
            new_centroids.append(average(pList))

        centroids = new_centroids

    return centroids, assignments, totalCost

    # ### END CODE HERE ###