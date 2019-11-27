 #!/usr/bin/env python
from pddlsim.executors.plan_dispatch import PlanDispatcher
from pddlsim.local_simulator import LocalSimulator
from pddlsim.executors.executor import Executor
import pddlsim.planner as planner

from collections import Counter, defaultdict
from random import choice
import sys
import os

# The class define the deterministic planner (best for deterministic worlds).
class PlanDispatcher(Executor):
    """
    docstring for PlanDispatcher.
    The class define and run the planner for deterministic worlds.
    NOTE: use inside try and catch as it will fail in non deterministic
          worlds.
    """
    def __init__(self):
        super(PlanDispatcher, self).__init__()

    def initialize(self,services):
        self.path = planner.make_plan(services.pddl.domain_path,services.pddl.problem_path)

    def next_action(self):
        if len(self.path) > 0:
            return self.path.pop(0).lower()
        return None


class ReinforcementLearning(object):
    """docstring for ReinforcementLearning"""
    def __init__(self, policy):
        self.successor = None
        self.Q = policy

    def initialize(self,services):
        self.services = services

    def next_action(self):
        raise NotImplementedError

    def get_max_Q(self, state, options):
        # Init the return option with the first option (there can be negative score in
        #   Q meaning there is no definitive first value for the score).
        ret_option = options[0]
        ret_option_score = self.Q[' '.join((state, ret_option))]

        # Search for the option with the max Q score.
        for cur_option in options[1:]:
            if self.Q[' '.join((state, cur_option))] > ret_option_score:
                ret_option_score = self.Q[' '.join((state, cur_option))]
                ret_option = cur_option

        return ret_option, ret_option_score


class QLearningEmpower(ReinforcementLearning):
    """docstring for QLearningEmpower"""
    def __init__(self, policy, write):
        super(QLearningEmpower, self).__init__(policy)
        self.write = write
        self.num_of_steps = 0

    def initialize(self, services):
        super(QLearningEmpower, self).initialize(services)

        self.s_a = "" # The current state_action (None - start position)
        self.alpha = 0.5
        self.gamma = 0.2

        self.goals = self.services.parser.goals[0].parts
        self.reached_goals = 0
        self.previous_state = ""

        # True - get the option acording to Q, False - random choice.
        probability = [(7, 3), (6, 4), (4, 6), (3, 7)]

        # if the iteration grater ther 200 pick one randomly else take the corrisponding one.
        if (int(self.Q["~probability~"]) // 50) > 4:
            x, y = probability[choice(range(4))]
        else:
            x, y = probability[int(self.Q["~probability~"]) // 50]

        self.choose_option = [False] * x + [True] * y
        self.Q["~probability~"] += 1

    def next_action(self):
        # If the goal has been reached finish the program.
        if self.services.goal_tracking.reached_all_goals():
            # Since we finish there is no next action thus the update is only the reward.
            self.Q[self.s_a] += self.reward()
            self.write()
            return None

        # Get the state, the valid actions and the max current Q(s,a).
        options = self.services.valid_actions.get()
        state = str(self.services.perception.state)
        max_Q, max_Q_score = self.get_max_Q(state, options)

        # Update the Q(s,a) of the previous state.
        if self.s_a:
            max_Q_score = self.gamma * (max_Q_score - self.Q[self.s_a])
            self.Q[self.s_a] += self.alpha * (self.reward(len(options)) + max_Q_score)

            # Write to the policy file every 30 steps.
            if not (self.num_of_steps % 30):
                self.write()

        if choice(self.choose_option):
            option = max_Q
        else:
            option = choice(options)

        self.previous_state = self.s_a.rsplit(" (", 1)[0]
        self.s_a = ' '.join((state, option))
        self.num_of_steps += 1
        return option

    def reached_a_goal(self):
        state = self.services.perception.state
        cur_reached_goals = 0

        for goal in self.goals:
            if self.services.parser.test_condition(goal, state):
                cur_reached_goals += 1

        if cur_reached_goals > self.reached_goals:
            self.reached_goals = cur_reached_goals
            return True

        return False

    def reward(self, num_of_options=0):
        if self.services.goal_tracking.reached_all_goals():
            return 300

        if self.previous_state == str(self.services.perception.state):
            return -200

        ret_reward = 2 * num_of_options
        if self.reached_a_goal():
            ret_reward += 150

        return ret_reward


class SARSAEmpower(QLearningEmpower):
    """docstring for SARSAEmpower"""
    def __init__(self, policy, write):
        super(SARSAEmpower, self).__init__(policy, write)

    def initialize(self, services):
        super(SARSAEmpower, self).initialize(services)

    def next_action(self):
        # If the goal has been reached finish the program.
        if self.services.goal_tracking.reached_all_goals():
            # Since we finish there is no next action thus the update is only the reward.
            self.Q[self.s_a] += self.reward()
            self.write()
            return None

        # Get the state, the valid actions and the max current Q(s,a).
        options = self.services.valid_actions.get()
        state = str(self.services.perception.state)

        if choice(self.choose_option):
            option, option_score = self.get_max_Q(state, options)
        else:
            option = choice(options)
            option_score = self.Q[' '.join((state, option))]

        # Update the Q(s,a) of the previous state.
        if self.s_a:
            option_score = self.gamma * (option_score - self.Q[self.s_a])
            self.Q[self.s_a] += self.alpha * (self.reward(len(options)) + option_score)

            # Write to the policy file every 30 steps.
            if not (self.num_of_steps % 30):
                self.write()

        self.previous_state = self.s_a.rsplit(" (", 1)[0]
        self.s_a = ' '.join((state, option))
        self.num_of_steps += 1
        return option


class ReinforcementLearningExecution(ReinforcementLearning):
    """docstring for ReinforcementLearningExecution"""
    def __init__(self, policy):
        super(ReinforcementLearningExecution, self).__init__(policy)

    def initialize(self, services):
        super(ReinforcementLearningExecution, self).initialize(services)

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            return None

        # Get the state and the valid actions.
        options = self.services.valid_actions.get()

        # Return the optimal (max Q(s,a)) action by using Q
        option, _ = self.get_max_Q(str(self.services.perception.state), options)
        return option


class Controller(object):
    """docstring for Controller"""
    def __init__(self, domain_file, problem_file, policy_file):
        self.domain_file = domain_file
        self.problem_file = problem_file

        def write_policy():
            def ret_func():
                file = open(policy_file, mode='w')

                # Write "info".
                file.write(self.data["info"]["best"] + '\n')
                file.write(str(self.data["info"]["score"]) + '\n')
                file.write(str(self.data["info"]["run"]) + '\n')
                file.write("---\n")

                # Write "Q-Learning" policy.
                for value in self.data["Q-Learning"]:
                    file.write("{} {}\n".format(value, str(self.data["Q-Learning"][value])))

                file.write("---\n")

                # Write "SARSA" policy.
                for value in self.data["SARSA"]:
                    file.write("{} {}\n".format(value, str(self.data["SARSA"][value])))

                file.close()

            return ret_func

        self.write = write_policy()
        self.read_policy(policy_file)

    def read_policy(self, policy_file):
        # Load the policy file if it exists.
        if os.path.isfile(policy_file):
            self.data = {"info":{}, "Q-Learning":Counter(), "SARSA":Counter()}
            file = open(policy_file, mode='r')

            # Set "info".
            self.data["info"]["best"] = file.readline().strip()
            self.data["info"]["score"] = int(file.readline().strip())
            self.data["info"]["run"] = int(file.readline().strip())
            file.readline().strip() # Read the first "---" separator.

            # Set "Q-Learning" and "SARSA" policies.
            policy = self.data["Q-Learning"]

            for line in file:
                line = line.strip().rsplit(' ', 1)
                if line[0] == "---":
                    policy = self.data["SARSA"]


                else:
                    value = line[1]
                    if str.isdigit(value.replace('.', '', 1)):
                        policy[line[0]] = float(line[1])

            file.close()

        # There is no policy file. the program initialize self.data.
        else:
            self.data = {"info":{"best":"Q-Learning", "score":0, "run":-1}, "Q-Learning":Counter(), "SARSA":Counter()}
            self.write()

    def run(self, phase):
        # Learning phase
        if phase == "-L":
            self.data["info"]["run"] += 1
            self.write()

            if self.data["info"]["run"] == 0:
                try:
                    report = LocalSimulator().run(sys.argv[2], sys.argv[3], PlanDispatcher())
                    self.set_record(report, "Planner")
                except Exception as e:
                    return

            elif (self.data["info"]["run"] > 400) and (self.data["info"]["run"] % 20 == 0):
                report = LocalSimulator().run(sys.argv[2], sys.argv[3], ReinforcementLearningExecution(self.data["Q-Learning"]))
                self.set_record(report, "Q-Learning")

            elif (self.data["info"]["run"] > 400) and (self.data["info"]["run"] % 20 == 1):
                report = LocalSimulator().run(sys.argv[2], sys.argv[3], ReinforcementLearningExecution(self.data["SARSA"]))
                self.set_record(report, "sarsa")

            # The run is odd number.
            elif (self.data["info"]["run"] % 2):
                print LocalSimulator().run(sys.argv[2], sys.argv[3], QLearningEmpower(self.data["Q-Learning"], self.write))
                # print LocalSimulator().run(sys.argv[2], sys.argv[3], SARSAEmpower(self.data["SARSA"], self.write))

            # The run is even number.
            else:
                print LocalSimulator().run(sys.argv[2], sys.argv[3], SARSAEmpower(self.data["SARSA"], self.write))
                # print LocalSimulator().run(sys.argv[2], sys.argv[3], QLearningEmpower(self.data["Q-Learning"], self.write))

        # Execution phase
        elif phase == "-E":
            if not self.data["info"]["best"]:
                print "Please run learning phase first (-L)"

            elif self.data["info"]["best"] == "Planner":
                executer = PlanDispatcher()

            elif self.data["info"]["best"] == "Q-Learning":
                executer = ReinforcementLearningExecution(self.data["Q-Learning"])

            else: # self.data["info"]["best"] == "sarsa"
                executer = ReinforcementLearningExecution(self.data["SARSA"])

            print "Running execution phase"
            print LocalSimulator().run(sys.argv[2], sys.argv[3], executer)
            print "\nThe program used {} for the given problem".format(self.data["info"]["best"])

        # Learning/Execution tag error
        else:
            print "Invalid phase option (valid options are '-L' and '-E')"

    def set_record(self, report, lerner_name):
        if (not self.data["info"]["score"] and report.total_actions) or (report.total_actions < self.data["info"]["score"]):
            self.data["info"]["best"] = lerner_name
            self.data["info"]["score"] = report.total_actions
            self.write()
        print report


# Define the main function
def main():
    domain_file = sys.argv[2]
    problem_file = sys.argv[3]
    policy_file = "_policy.txt"

    # Get the problem name for the policy name.
    file = open(problem_file, mode='r')
    line = file.readline().strip()
    while "define" not in line:
        line = file.readline().strip()

    line = line.split(')',1)[0].rsplit(' ', 1)[1]
    policy_file = '_' + line + policy_file
    file.close()

    # Get the domain name for the policy name.
    file = open(domain_file, mode='r')
    line = file.readline().strip()
    while "define" not in line:
        line = file.readline().strip()

    line = line.split(')',1)[0].rsplit(' ', 1)[1]
    policy_file = line + policy_file
    file.close()

    controller = Controller(domain_file, problem_file, policy_file)
    controller.run(sys.argv[1])


if __name__ == '__main__':
    main()