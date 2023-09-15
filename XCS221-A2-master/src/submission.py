from typing import Callable, List, Set

import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model


class SegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        pass
        # ### START CODE HERE ###
        return self.query
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        return len(state) == 0
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        result = []
        if not self.isEnd(state):
            for i in range(len(state), 0, -1):
                action = state[:i]
                cost = self.unigramCost(action)
                rem_text = state[len(action):]
                result.append((action, rem_text, cost))
        return result
        # ### END CODE HERE ###


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ""

    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # ### START CODE HERE ###
    words = ' '.join(ucs.actions)
    return words
    # ### END CODE HERE ###


############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost


class VowelInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        queryWords: List[str],
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        pass
        # ### START CODE HERE ###
        return self.queryWords[0], 0
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        return state[1] == len(self.queryWords)-1
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        result = []
        index = state[1] + 1
        choices = self.possibleFills(self.queryWords[index]).copy()
        if len(choices) == 0:
            choices.add(self.queryWords[index])
        for action in choices:
            cost = self.bigramCost(state[0], action)
            result.append((action, (action, index), cost))
        return result
        # ### END CODE HERE ###


def insertVowels(
    queryWords: List[str],
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    pass
    # ### START CODE HERE ###
    if len(queryWords) == 0:
        return ''
    else:
        queryWords.insert(0, wordsegUtil.SENTENCE_BEGIN)
    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    words = ' '.join(ucs.actions)
    return words
    # ### END CODE HERE ###


############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem


class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        query: str,
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        pass
        # ### START CODE HERE ###
        return self.query, wordsegUtil.SENTENCE_BEGIN
        # ### END CODE HERE ###

    def isEnd(self, state) -> bool:
        pass
        # ### START CODE HERE ###
        if len(state[0]) == 0:
            return True
        return False
        # ### END CODE HERE ###

    def succAndCost(self, state):
        pass
        # ### START CODE HERE ###
        result = []
        for i in range(1, len(state[0]) + 1):
            pre_word = state[0][:i]
            rem_word = state[0][i:]
            choices = self.possibleFills(pre_word).copy()
            for item in choices:
                cost = self.bigramCost(state[1], item)
                result.append((item, (rem_word, item), cost))
        return result
        # ### END CODE HERE ###


def segmentAndInsert(
    query: str,
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    if len(query) == 0:
        return ""

    # ### START CODE HERE ###
    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    words = ' '.join(ucs.actions)
    return words
    # ### END CODE HERE ###


############################################################

if __name__ == "__main__":
    shell.main()
