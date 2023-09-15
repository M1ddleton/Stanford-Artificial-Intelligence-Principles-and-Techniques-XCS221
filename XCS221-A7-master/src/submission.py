import collections, sys, os
from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a():
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    # ### START CODE HERE ###
    return Implies(And(Summer, California), Not(Rain))
    # ### END CODE HERE ###

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b():
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    # ### START CODE HERE ###
    return Equiv(Wet, Or(Rain, Sprinklers))
    # ### END CODE HERE ###

# Sentence: "Either it's day or night (but not both)."
def formula1c():
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    # ### START CODE HERE ###
    return Or(And(Day, Not(Night)), And(Not(Day), Night))
    # ### END CODE HERE ###

############################################################
# Problem 2: first-order logic

# Sentence: "Every person has a mother."
def formula2a():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Mother(x, y): return Atom('Mother', x, y)  # whether x's mother is y

    # Note: You do NOT have to enforce that the mother is a "person"
    # ### START CODE HERE ###
    return Forall('$x', Exists('$y', Implies(Person('$x'), Mother('$x', '$y'))))
    # ### END CODE HERE ###

# Sentence: "At least one person has no children."
def formula2b():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y

    # Note: You do NOT have to enforce that the child is a "person"
    # ### START CODE HERE ###
    return Exists('$x', Not(Exists('$y', Implies(Person('$x'), Child('$x', '$y')))))
    # ### END CODE HERE ###

# Return a formula which defines Daughter in terms of Female and Child.
# See parentChild() in examples.py for a relevant example.
def formula2c():
    # Predicates to use:
    def Female(x): return Atom('Female', x)            # whether x is female
    def Child(x, y): return Atom('Child', x, y)        # whether x has a child y
    def Daughter(x, y): return Atom('Daughter', x, y)  # whether x has a daughter y
    # ### START CODE HERE ###
    return Forall('$x', Forall('$y', Equiv(And(Child('$x', "$y"), Female('$y')), Daughter('$x', "$y"))))
    # ### END CODE HERE ###

# Return a formula which defines Grandmother in terms of Female and Parent.
# Note: It is ok for a person to be her own parent
def formula2d():
    # Predicates to use:
    def Female(x): return Atom('Female', x)                  # whether x is female
    def Parent(x, y): return Atom('Parent', x, y)            # whether x has a parent y
    def Grandmother(x, y): return Atom('Grandmother', x, y)  # whether x has a grandmother y
    # ### START CODE HERE ###
    return Forall('$x', Forall('$z', Equiv(
        And(Exists('$y', And(Parent('$x', '$y'), Parent('$y', '$z'))), Female('$z')), Grandmother('$x', '$z')
    )))
    # ### END CODE HERE ###

############################################################
# Problem 3: Liar puzzle

# Facts:
# 0. John: "It wasn't me!"
# 1. Susan: "It was Nicole!"
# 2. Mark: "No, it was Susan!"
# 3. Nicole: "Susan's a liar."
# 4. Exactly one person is telling the truth.
# 5. Exactly one person crashed the server.
# Query: Who did it?
# This function returns a list of 6 formulas corresponding to each of the
# above facts.
# Hint: You might want to use the Equals predicate, defined in logic.py.  This
# predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False iff x is not equal to y.
def liar():
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    john = Constant('john')
    susan = Constant('susan')
    nicole = Constant('nicole')
    mark = Constant('mark')
    formulas = []
    # We provide the formula for fact 0 here.
    formulas.append(Equiv(TellTruth(john), Not(CrashedServer(john))))
    # You should add 5 formulas, one for each of facts 1-5.
    # ### START CODE HERE ###
    formulas.append(Equiv(TellTruth(susan), CrashedServer(nicole)))
    formulas.append(Equiv(TellTruth(mark), CrashedServer(susan)))
    formulas.append(Equiv(TellTruth(nicole), Not(TellTruth(susan))))
    formulas.append(Exists('$x', And(TellTruth('$x'), Forall('$y', Implies(TellTruth('$y'), Equals('$x', '$y'))))))
    formulas.append(Exists('$x', And(CrashedServer('$x'), Forall('$y', Implies(CrashedServer('$y'), Equals('$x', '$y'))))))
    # ### END CODE HERE ###
    query = CrashedServer('$x')
    return (formulas, query)

############################################################
# Problem 4: Odd and even integers

# Return the following 6 laws:
# 0. Each number $x$ has exactly one successor, which is not equal to $x$.
# 1. Each number is either even or odd, but not both.
# 2. The successor number of an even number is odd.
# 3. The successor number of an odd number is even.
# 4. For every number $x$, the successor of $x$ is larger than $x$.
# 5. Larger is a transitive property: if $x$ is larger than $y$ and $y$ is
#    larger than $z$, then $x$ is larger than $z$.
# Query: For each number, there exists an even number larger than it.
def ints():
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y
    # Note: all objects are numbers, so we don't need to define Number as an
    # explicit predicate.
    # Note: pay attention to the order of arguments of Successor and Larger.
    # Populate |formulas| with the 6 laws above and set |query| to be the
    # query.
    # Hint: You might want to use the Equals predicate, defined in logic.py.  This
    # predicate is used to assert that two objects are the same.
    formulas = []
    query = None
    # ### START CODE HERE ###
    formulas.append(Forall('$x', Exists('$y', AndList([Successor('$x', '$y'),
                                                       Not(Equals('$x', '$y')),
                                                       Forall('$z', Implies(Successor('$x', '$z'),
                                                                            Equals('$y', '$z')))]))))
    formulas.append(Forall('$x', Or(And(Even('$x'), Not(Odd('$x'))),
                                    And(Odd('$x'), Not(Even('$x')))
                                    )))
    formulas.append(Forall('$x', Forall('$y', Implies(And(Even('$x'), Successor('$x', "$y")), Odd('$y'))
                                        )))
    formulas.append(Forall('$x', Forall('$y', Implies(And(Odd('$x'), Successor('$x', '$y')), Even('$y'))
                                        )))
    formulas.append(Forall('$x', Forall('$y', Implies(
        Successor('$x', '$y'),
        Larger('$y', '$x')
    ))))
    formulas.append(Forall('$x', Forall('$y', Forall('$z',
        Implies(
            And(Larger('$x', '$y'), Larger('$y', '$z')),
                Larger('$x', '$z')
                                )))))
    # ### END CODE HERE ###
    query = Forall('$x', Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)

############################################################
# Problem 5: semantic parsing (extra credit)
# Each of the following functions should return a GrammarRule.
# Look at createBaseEnglishGrammar() in nlparser.py to see what these rules should look like.
# For example, the rule for 'X is a Y' is:
#     GrammarRule('$Clause', ['$Name', 'is', 'a', '$Noun'],
#                 lambda args: Atom(args[1].title(), args[0].lower()))
# Note: args[0] corresponds to $Name and args[1] corresponds to $Noun.
# Note: by convention, .title() should be applied to all predicates (e.g., Cat).
# Note: by convention, .lower() should be applied to constant symbols (e.g., garfield).

from nlparser import GrammarRule

def createRule1():
    # Return a GrammarRule for 'every $Noun $Verb some $Noun'
    # Note: universal quantification should be outside existential quantification.
    pass
    # ### START CODE HERE ###
    return GrammarRule('$Clause', ['every', '$Noun', '$Verb', 'some', '$Noun'],
                       lambda args: Forall('$x', Forall('$y', Implies(
                           And(Atom(args[0].title(), '$x'), Atom(args[1].title(), '$y')),
                           Exists('$z', And(Atom(args[2].title(), '$y'), Atom(args[1].title(), args[0].lower(), '$y')))
                       ))))
    # ### END CODE HERE ###

def createRule2():
    # Return a GrammarRule for 'there is some $Noun that every $Noun $Verb'
    pass
    # ### START CODE HERE ###
    return GrammarRule('$Clause', ['there', 'is', 'some', '$Noun', 'that', 'every', '$Noun', '$Verb'],
                       lambda args: Exists('$x', Forall('$y', Forall('$z', Implies(
                                               And(Atom(args[1].title(), '$y'), Atom(args[2].title(), '$z')),
                                               Atom(args[0].title(), '$x')
                                           )))))
    # ### END CODE HERE ###

def createRule3():
    # Return a GrammarRule for 'if a $Noun $Verb a $Noun then the former $Verb the latter'
    pass
    # ### START CODE HERE ###
    return GrammarRule('$Clause', ['if', 'a', '$Noun', '$Verb', 'a', '$Noun', 'then', 'the', 'former', '$Verb', 'the',
                                   'latter'],
                       lambda args: Forall('$x', Forall('$y', Forall('$z',
                                                                     Implies(And(Atom(args[0].title(), '$x'),
                                                                                      Atom(args[1].title(), '$y'),
                                                                                      Atom(args[2].title(), '$z'))
                                                                       )))))
    # ### END CODE HERE ###
