import util, math, random
from collections import defaultdict
from util import ValueIteration
from typing import List, Callable, Tuple, Any


############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        pass
        # ### START CODE HERE ###
        result = []
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state
        if deckCardCounts is None:
            return []
        ##################################################################################################################

        elif action == 'Quit' or sum(deckCardCounts) == 0:
            next_state = (0, None, None)
            next_reward = totalCardValueInHand
            if totalCardValueInHand > self.threshold:
                next_reward = 0
            result.append((next_state, 1.0, next_reward))
        ###################################################################################################################

        elif action == 'Take':
            if nextCardIndexIfPeeked is not None:
                deck_card_count_list = list(deckCardCounts)
                deck_card_count_list[nextCardIndexIfPeeked] -= 1
                new_value_in_hand = self.cardValues[nextCardIndexIfPeeked] + totalCardValueInHand
                if new_value_in_hand > self.threshold:
                    next_state = (new_value_in_hand, None, None)
                elif sum(deck_card_count_list) == 0:
                    next_state = (new_value_in_hand, None, None)
                    next_reward = new_value_in_hand
                else:
                    next_state = (new_value_in_hand, None, tuple(deck_card_count_list))
                result.append((next_state, 1.0, 0))
            else:
                for index, item in enumerate(deckCardCounts):
                    if item > 0:
                        deck_card_count_list = list(deckCardCounts)
                        next_prob = float(item) / sum(deckCardCounts)
                        deck_card_count_list[index] -= 1
                        new_value_in_hand = self.cardValues[index] + totalCardValueInHand
                        next_reward = 0
                        if new_value_in_hand > self.threshold:
                            next_state = (new_value_in_hand, None, None)
                        elif sum(deck_card_count_list) == 0:
                            next_state = (new_value_in_hand, None, None)
                            next_reward = new_value_in_hand
                        else:
                            next_state = (new_value_in_hand, None, tuple(deck_card_count_list))
                        result.append((next_state, next_prob, next_reward))
        ######################################################################################################

        elif action == 'Peek':
            if nextCardIndexIfPeeked is not None:
                return []
            for index, item in enumerate(deckCardCounts):
                if item > 0:
                    next_prob = float(item) / sum(deckCardCounts)
                    next_state = (totalCardValueInHand, index, deckCardCounts)
                    next_reward = - self.peekCost
                    result.append((next_state, next_prob, next_reward))
        return result
        # ### END CODE HERE ###

    def discount(self):
        return 1


############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        pass
        # ### START CODE HERE ###
        V_opt = 0.0
        if newState is not None:
            V_opt = max([self.getQ(newState, new_action) for new_action in self.actions(newState)])
        Q_opt = self.getQ(state, action)
        adjust = - self.getStepSize() * (Q_opt - (reward + self.discount * V_opt))
        for item in self.featureExtractor(state, action):
            key, value = item
            self.weights[key] = self.weights[key] + adjust * value
        # ### END CODE HERE ###


# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)


def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    pass
    # ### START CODE HERE ###
    mdp.computeStates()
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor, 0.2)

    rewards = util.simulate(mdp, rl, numTrials=30000, maxIterations=1000, verbose=False, sort=False)
    rl.explorationProb = 0.0
    vi = ValueIteration()
    vi.solve(mdp)
    mismatches = 0
    for state in mdp.states:
        if rl.getAction(state) != vi.pi[state]:
            mismatches += 1
    print(mismatches, len(mdp.states))
    # ### END CODE HERE ###


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # ### START CODE HERE ###
    features = []
    f_key = (action, total)
    f_value = 1
    features.append((f_key, f_value))
    if counts is not None:
        counts_list = list(counts)
        for index, item in enumerate(counts):
            f_key = (action, index, item)
            f_value = 1
            features.append((f_key, f_value))
            if item > 0:
                counts_list[index] = 1
        f_key = (action, tuple(counts_list))
        f_value = 1
        features.append((f_key, f_value))
    return features
    # ### END CODE HERE ###


############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)


def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    pass
    # ### START CODE HERE ###
    original_mdp.computeStates()
    vi = ValueIteration()
    vi.solve(original_mdp)

    rl = util.FixedRLAlgorithm(vi.pi.copy())
    reward = util.simulate(modified_mdp, rl, numTrials=10000, maxIterations=1000, verbose=False, sort=False)
    rl.explorationProb = 0.0
    modified_mdp.computeStates()
    rl = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor, 0.2)
    reward = util.simulate(modified_mdp, rl, numTrials=10000, maxIterations=1000, verbose=False, sort=False)

    # print(reward)
    # ### END CODE HERE ###


############################################################
# Problem 5: Modeling sea level rise

class SeaLevelRiseMDP(util.MDP):

    def __init__(self, initial_infra: int, n_years: int, init_budget: int, disaster: bool = False, discount: int = 1,
                 failure_cost=-1000):
        """
        
        initial_infra: initial state of infrastructure
        n_years: how many years to run the simulation
        init_budget: initial amount in budget
        disaster: whether to include a small probability of catastrophic disaster each step
        discount: discount factor
        rate: how quickly the sea level rises (default 10, should almost never change, can make 9 to double the sea level rise)
        
        """
        self.initial_sea = 0
        self.initial_infra = initial_infra
        self.n_years = n_years
        self.init_budget = init_budget
        self.start_year = 2000
        self.end_year = self.start_year + self.n_years
        self.disaster = disaster
        self.disc = discount
        self.failure_cost = failure_cost

    # Return the start state.
    # Look closely at this function to see an example of state representation for this simulation.
    # Each state is a tuple with 4 elements:
    #   -- The first element of the tuple is the current year in the simulation.
    #   -- The second element of the tuple is current amount in the budget.
    #   -- The third element is the amount of built infrastructure to date
    #   -- The fourth element is the current sea level.
    def startState(self) -> Tuple:
        return (self.start_year, self.init_budget, self.initial_infra, self.initial_sea)

        # Return set of actions possible from |state|.

    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state: Tuple) -> List[str]:
        return ['Invest', 'Wait']  # the city can either invest in infrastructure or wait

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after the max number of years have been reached or disaster happens)
    #   by a negative year.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    # * When the sea has overtaken the infrastructure in the current state, immediately transition to the end state and incur the failure_cost
    #   reward punishment without updating the infrastructure, budget, or sea level. 
    # * If disaster strikes in one of the next states, still update the infrastructure, the budget, and sea level as normal and
    #   transition to the end state with the appropriate negative reward. 
    # * At the end of the simulation (the final year is reached), do not update the budget, infrastructure ,or sea level. However if the
    #   sea has overtaken the infrastructure, treat it as a disaster and include the negative reward when transitioning to the end state.
    # * When self.disc == 1, this means no discount is being applied and no reward should be given for intermediate states
    # * Note: infrastructure and budget can be negative. Number of years, sea level, and discount should not
    def succAndProbReward(self, state: Tuple, action: str) -> List[Tuple]:

        year, budget, infra, s_level = state

        # simulation is over if the number of years to run is exceeded or if disaster occurs
        if (year < 0):
            return []

        results = []  # The possible (newState, prob, reward) triples
        # if the total number of years to run the simulation has passed
        if year == self.end_year:
            # indicate simulation over, don't update budget or infra or sea level
            # if sea level has passed, then still disaster
            reward = self.failure_cost if infra <= s_level else budget
            newState = (-1, budget, infra, s_level)
            # whatever budget remains is given as reward, or punishment if disaster occurs
            results.append((newState, 1.0, reward))
        else:
            # the three possible sea level rises: low, medium, high
            # medium rise is twice as likely as low or high
            sea_states = {1: 0.25, 2: 0.5, 3: 0.25}
            # check to see if we're calculating if there's a chance of disaster
            # at each time step. This is akin to the possibility of a hurricane hitting
            if self.disaster:
                disaster = .1 / (infra - s_level) if infra > s_level else 1
            else:
                # if we're not checking for chance of disaster, then disaster only
                # occurs if the sea level is greater than the city.
                # # This is akin to the sea flooding the city
                disaster = 0 if infra > s_level else 1
            # loop through each possible next sea level rise
            if disaster == 1.0:
                # immediately break with full probability
                results.append(((-1, budget, infra, s_level), 1.0, self.failure_cost))
            else:
                for rise, prob in sea_states.items():
                    # if the city chooses to wait
                    if action == "Wait":
                        # they get an extra 2 budget, but infra doesn't change
                        n_budget = budget + 2
                        n_infra = infra
                    # if the city chooses to invest
                    elif action == 'Invest':
                        # they lose 2 budget but gain 3 infrastructure
                        n_budget = budget - 2
                        n_infra = infra + 3
                    else:
                        raise Exception('Invalid action: %s' % action)

                    # calculate the probability of disaster
                    prob_d = prob * disaster
                    # if disaster occurs, city loses and simulation ends
                    state = (self.end_year, n_budget, n_infra, s_level + rise)
                    # disaster incurs a punishment of -1000, set year to -1 to indicate simulation over if disaster can happen
                    results.append(
                        ((-1, n_budget, n_infra, s_level + rise), prob_d, self.failure_cost)) if disaster > 0 else None
                    # calculate the probability of no disaster
                    prob_n = prob * (1 - disaster)
                    # if disaster doesn't occur, 10 more years pass, and the sea level rises by given amount
                    state = (year + 10, n_budget, n_infra, s_level + rise)
                    # if doing discount, return the current budget as the reward,
                    # else, no reward at this timestep
                    results.append((state, prob_n, 0)) if self.disc == 1 else results.append((state, prob_n, n_budget))

        return results

    def discount(self):
        return self.disc


############################################################
# Problem 5a: Comparing model time horizons

# This is a helper function for comparing the predicted optimal 
# actions for 2 MDPs of varying timescales of simulation

short_time = SeaLevelRiseMDP(initial_infra=12, n_years=40, init_budget=3)
long_time = SeaLevelRiseMDP(initial_infra=12, n_years=100, init_budget=3)

discounted = SeaLevelRiseMDP(initial_infra=14, n_years=100, init_budget=5, discount=0.7, disaster=True)
no_discount = SeaLevelRiseMDP(initial_infra=14, n_years=100, init_budget=5, disaster=True)


def sampleKTrajectories(mdp: SeaLevelRiseMDP, val: util.ValueIteration):
    invs, wats = 0, 0
    for n in range(1000):
        traj = util.sample_trajectory(mdp, val)
        invs += traj.count('Invest')
        wats += traj.count('Wait')

    print(
        f"\nVIter with MDP -> year:{mdp.startState()[0]}, budget:{mdp.startState()[1]}, infra:{mdp.startState()[2]}, sealevel:{mdp.startState()[3]} n_years:{mdp.n_years}, & discount:{mdp.discount()}")
    print(f"  *  total invest states: {invs} total wait states: {wats}")
    print(f"  *  ratio of invest to wait states: {invs / wats}\n")


def sampleKRLTrajectories(mdp: SeaLevelRiseMDP, rl: QLearningAlgorithm):
    invs, wats = 0, 0
    for n in range(1000):
        traj = util.sample_RL_trajectory(mdp, rl)
        invs += traj.count('Invest')
        wats += traj.count('Wait')

    print(
        f"\nRL with MDP -> year:{mdp.startState()[0]}, budget:{mdp.startState()[1]}, infra:{mdp.startState()[2]}, sealevel:{mdp.startState()[3]} n_years:{mdp.n_years}, & discount:{mdp.discount()}")
    print(f"  *  total invest states: {invs} total wait states: {wats}")
    print(f"  *  ratio of invest to wait states: {invs / wats}\n")


# This is a helper function for both 5b and 5d. This function runs 
# ValueIteration, then simulates various trajectories through the MDP
# and compares the frequency of various optimal actions.
def compare_MDP_Strategies(mdp1: SeaLevelRiseMDP, mdp2: SeaLevelRiseMDP):
    # first, run value iteration on the mdp1 timescale MDP
    v_mdp1 = util.ValueIteration()
    v_mdp1.solve(mdp1, .0001)
    # then, run value iteration on the mdp2 timescale MDP
    v_mdp2 = util.ValueIteration()
    v_mdp2.solve(mdp2, .0001)
    # sample 1000 different trajectories through the MDP and
    # count the number of times the government waits versus invests

    sampleKTrajectories(mdp1, v_mdp1)
    sampleKTrajectories(mdp2, v_mdp2)



############################################################
# Problem 5d: Exploring how policies transfer

# This is a helper function for comparing the predicted optimal 
# actions for 2 MDPs of varying timescales of simulation
high_cost = SeaLevelRiseMDP(initial_infra=50, n_years=100, init_budget=3, failure_cost=-10000, disaster=True)
low_cost = SeaLevelRiseMDP(initial_infra=50, n_years=100, init_budget=3, failure_cost=-10, disaster=True)


def compare_changed_SeaLevelMDP(orig_mdp: SeaLevelRiseMDP, modified_mdp: SeaLevelRiseMDP):
    # first, look at how expected reward changes transfering policies between MDPs
    print('\n--------- Part 1. ------------')
    # run ValueIteration on original MDP
    v_iter = ValueIteration()
    v_iter.solve(orig_mdp)
    # simulate ValueIteration of original MDP on the modified one
    fixed_rl = util.FixedRLAlgorithm(v_iter.pi)
    fixed_rl_rewards = util.simulate(modified_mdp, fixed_rl, numTrials=30000)
    exp_reward = sum(fixed_rl_rewards) / float(len(fixed_rl_rewards))

    print(
        f"\n Pi of Original MDP -> year:{orig_mdp.startState()[0]}, budget:{orig_mdp.startState()[1]}, infra:{orig_mdp.startState()[2]}, sealevel:{orig_mdp.startState()[3]} n_years:{orig_mdp.n_years}, & failure_cost:{orig_mdp.failure_cost}")
    print(f"   *   Expected reward on Original MDP: {v_iter.V[orig_mdp.startState()]}")

    print('\n----------- Part 2. ------------')
    print(
        f"\n Pi of Original MDP -> year:{orig_mdp.startState()[0]}, budget:{orig_mdp.startState()[1]}, infra:{orig_mdp.startState()[2]}, sealevel:{orig_mdp.startState()[3]} n_years:{orig_mdp.n_years}, & failure_cost:{orig_mdp.failure_cost}")
    print(f"   *   Expected reward on Modified MDP: {exp_reward}")
    print(
        f"   *   Difference in expected reward between original and modified MDP: {exp_reward - v_iter.V[orig_mdp.startState()]}")

    # next, compare how the action choices of the different MDP policies change
    print('\n----------- Part 3. ------------')

    # run value iteration on modified MDP
    v_mod = ValueIteration()
    v_mod.solve(modified_mdp)

    # first, look at the distribution of optimal actions for the old MDP
    invs, wats = 0, 0
    for n in range(1000):
        traj = util.sample_trajectory(orig_mdp, v_iter)
        invs += traj.count('Invest')
        wats += traj.count('Wait')
    print(
        f"\nPi of Original MDP -> year:{orig_mdp.startState()[0]}, budget:{orig_mdp.startState()[1]}, infra:{orig_mdp.startState()[2]}, sealevel:{orig_mdp.startState()[3]} n_years:{orig_mdp.n_years}, & failure_cost:{orig_mdp.failure_cost}")
    print(f"  *  total invest states: {invs} total wait states: {wats}")
    print(f"  *  ratio of invest to wait states: {invs / wats}\n")
    # next, look at the distribution of optimal actions for new MDP
    invs, wats = 0, 0
    for n in range(1000):
        traj = util.sample_trajectory(modified_mdp, v_mod)
        invs += traj.count('Invest')
        wats += traj.count('Wait')
    print(
        f"\nPi of Modified MDP -> year:{modified_mdp.startState()[0]}, budget:{modified_mdp.startState()[1]}, infra:{modified_mdp.startState()[2]}, sealevel:{modified_mdp.startState()[3]} n_years:{modified_mdp.n_years}, & failure_cost:{modified_mdp.failure_cost}")
    print(f"  *  total invest states: {invs} total wait states: {wats}")
    print(f"  *  ratio of invest to wait states: {invs / wats}\n")
