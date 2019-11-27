# MULTI-PROGRAM_AGENT
## Background
The final project in 'Introduction to Intelligent, Knowledge-Based, and Cognitive Systems' course. The goal of the project is to create a general agent who will solve as many problems as possible in the most efficient and fast way it can.


## Dependencies
pddlsim==0.1.14.dev0


## The Atlantis model
The Atlantis model was originally intended to allow the agent to find a solution to a large number of problems that would happen to the agent when operating. The model does this by replacing the algorithm that the agent performs when it encounters a problem that the current algorithm cannot solve.


## The Multi-Program Model
The model will work in the same way when, instead of changing the algorithm during operation, it will learn in advance what the best algorithm is for the current problem during the learning process while testing a number of different algorithms. The agent start by checking PDDL built in planner before alternating between Q-Learning and SARSA.


## The algorithms
1. PDDL built in planner.
2. Q-Learning
3. SARSA

## Usage
To run the learning phase use:
```
python my_executive.py -L <domain_file> <problem_file>
```

For the execute phase use:
```
python my_executive.py -E <domain_file> <problem_file>
```

**NOTE: The learning phase need to be run a sufficient number of times to be effective.**
