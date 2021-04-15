import run_world as rw
import math
import numpy as np
import random

R_plus = 1.0
N_e = 300
num_rows = 3
num_cols = 4
num_moves = 4


def explore(u: float, n: int):
    '''
    determine the value of exploration function f given u and n
    '''
    if n < N_e:
        return R_plus

    return u


def get_prob(n_sa, n_sas, curr_state, dir_intended, next_state):
    '''
    Determine the transition probability based on counts. curr_state is s. dir_intended is +a+. 
    next_state is s'.
    '''

    if n_sa[curr_state[0]][curr_state[1]][dir_intended] > 0:
        assert n_sas[curr_state[0]][curr_state[1]][dir_intended][next_state[0]][next_state[1]] / n_sa[curr_state[0]][curr_state[1]][dir_intended] <= 1.0, \
            f"n_sas was {n_sas[curr_state[0]][curr_state[1]][dir_intended][next_state[0]][next_state[1]]} n_sa was {n_sa[curr_state[0]][curr_state[1]][dir_intended]}"

        return n_sas[curr_state[0]][curr_state[1]][dir_intended][next_state[0]][next_state[1]] / n_sa[curr_state[0]][curr_state[1]][dir_intended]

    return 0


def exp_utils(grid, utils, curr_state, n_sa, n_sas):
    '''
    Calculate the expected utilities sum_{s'}P(s'|s,a)V(s') for all four actions and return a list
    containing the four values.
    '''
    next_states = rw.get_next_states(grid, curr_state)
    vals = [0.0] * 4
    # TODO: are you supposed to use unique states?
    unique_next_states = set(next_states)
    for di in range(4):
        test = []
        for state in unique_next_states:
            test.append(get_prob(n_sa, n_sas, curr_state, di, state))
            vals[di] += get_prob(n_sa, n_sas, curr_state,
                                 di, state) * utils[state[0]][state[1]]

        assert math.isclose(sum(test), 1.0) or math.isclose(
            sum(test), 0.0), f"transition probabilities did not add up to 1.0: {test}"
    return vals


def optimistic_exp_utils(grid, utils, curr_state, n_sa, n_sas):
    '''
    Return the optimistic expected utilities f for all four actions and return a list containing the 
    four values
    '''

    if not rw.not_goal_and_wall(grid, curr_state):
        return [0,0,0,0]

    vals = []
    exp_utils_list = exp_utils(grid, utils, curr_state, n_sa, n_sas)
    
    for action, exp_util in enumerate(exp_utils_list):
        vals.append(
            explore(exp_util, n_sa[curr_state[0]][curr_state[1]][action]))

    return vals


def update_utils(world, grid, utils, n_sa, n_sas, gamma):
    '''
    Perform value iteration updates to the long-term expected utility estiminates until the estimates
    converge
    return: if any V value changed
    '''
    frontier = []
    util_updated = False

    for i in range(num_rows):
        for j in range(num_cols):
            if rw.not_goal_and_wall(grid, (i,j)):
                frontier.append((i, j))

    while(len(frontier) > 0):
        curr_state = frontier.pop()
        reward = get_reward_by_state(world, grid, curr_state)

        opt_exp_util_list = optimistic_exp_utils(
            grid, utils, curr_state, n_sa, n_sas)
        V = reward + gamma * max(opt_exp_util_list)

        if V > 2.0:
            # print(f"for state {curr_state}, V: {V}, util: {utils[curr_state[0]][curr_state[1]]}")
            # print(f"This should not happen. opt list", opt_exp_util_list)
            assert False

        # print(f"for state {curr_state}, V: {V}, util: {utils[curr_state[0]][curr_state[1]]}")

        # TODO: is this the right comparison
        if math.isclose(V, utils[curr_state[0]][curr_state[1]], rel_tol=0.0001):
            continue

        util_updated = True

        # value was different, update utils
        utils[curr_state[0]][curr_state[1]] = V

        # check if neighbouring states changed
        next_states = rw.get_next_states(grid, curr_state)
        for next_state in next_states:
            frontier.append(next_state)

    return util_updated

def utils_to_policy(grid, utils, n_sa, n_sas):
    '''
    Determine the optimal policy given to the current long-term utility value for each state.
    '''
    policy = np.array([[5] * num_cols for i in range(num_rows)]) # s
    # print(type(policy))
    # print(type(policy[0][0]))
    # print(type(grid))
    # print(type(grid[0][0]))
    for i, row in enumerate(utils):
        for j, val in enumerate(row):
            if not rw.not_goal_and_wall(grid, (i,j)):
                continue

            policy[i][j] = get_best_dir(grid, utils, (i,j), n_sa, n_sas)

    rw.pretty_print_policy(grid, policy)

def get_best_dir(grid, utils, curr_state, n_sa, n_sas):
    opt_exp_list = optimistic_exp_utils(
                grid, utils, curr_state, n_sa, n_sas)

    best_dirs = [0]
    best_dir_val = opt_exp_list[0]

    for k in range(1, 4):
        if best_dir_val < opt_exp_list[k]:
            best_dirs = [k]
            best_dir_val = opt_exp_list[k]
        elif best_dir_val == opt_exp_list[k]:
            best_dirs.append(k)

    return random.choice(best_dirs)


def get_reward_by_state(world, grid, state):
    if rw.is_goal(grid, state):
        return 1.0 if grid[state[0]][state[1]] == '1' else -1.0

    return rw.get_reward(world)

def state_action_pair_cond(grid, n_sa):
    mini = n_sa[0][0][0]
    for i, row in enumerate(n_sa):
        for j, col in enumerate(row):
            if not rw.not_goal_and_wall(grid, (i, j)):
                continue
            for val in col:
                mini = min(mini, val)

    return mini < N_e


def find_optimal_policy(world):
    # initialize variables
    curr_state = (0, 0)
    utils = [[0.0] * num_cols for i in range(num_rows)]  # s
    n_sa = np.array([[[0] * num_moves for i in range(num_cols)]
            for i in range(num_rows)])  # s by a
    n_sas = [[[[[0] * num_cols for i in range(num_rows)]
            for i in range(num_moves)] for i in range(num_cols)] for i in range(num_rows)]  # s by a by s'
    grid = rw.read_grid(world)

    iterations = 0
    util_updated = False

    # exit when each state-pair done at least N_e times and util stablizes
    while (state_action_pair_cond(grid, n_sa) or util_updated):
        iterations += 1
        # 2. decide on what the best action would be
        best_dir = get_best_dir(grid, utils, curr_state, n_sa, n_sas)

        # now make that move
        next_state = rw.make_move(grid, curr_state, best_dir, world)
        # print(f"choose to make move from {curr_state} in direction {best_dir}, next state is {next_state}")

        # 3. update n_sa and n_sas
        n_sa[curr_state[0]][curr_state[1]][best_dir] += 1
        n_sas[curr_state[0]][curr_state[1]
                             ][best_dir][next_state[0]][next_state[1]] += 1
        assert n_sa[curr_state[0]][curr_state[1]][best_dir] >= n_sas[curr_state[0]][curr_state[1]][best_dir][next_state[0]][next_state[1]], \
            f"After updating, n_sa[{curr_state[0]}][{curr_state[1]}][{best_dir}] was {n_sa[curr_state[0]][curr_state[1]][best_dir]}, {n_sas[curr_state[0]][curr_state[1]][best_dir][next_state[0]][next_state[1]]}"

        # 4. update utils based on new n_sa and n_sas
        util_updated = update_utils(world, grid, utils, n_sa, n_sas, rw.get_gamma(world))

        # reset this trial if we reach goal state
        if rw.is_goal(grid, next_state):
            curr_state = (0, 0)
        else:
            curr_state = next_state
        
        # print(state_action_pair_cond(grid, n_sa), util_updated)

    # debug print statements
    # print("n_sa:\n", n_sa)
    # print("num iterations:", iterations)

    print("Final utility values for", world, ":")
    print("[", end='')
    for i, row in enumerate(utils):
        print("[", end='')
        for j, val in enumerate(row):
            print("{:.3f}".format(val), end=' ')
        print("]") if i != 2 else print("]]")
     
    print("Optimal policy for", world, ":")
    utils_to_policy(grid, utils, n_sa, n_sas)
    print("")

def main():
    # dump_world_info("a4")
    # provided_helpers("lecture")
    # test_get_reward_by_state("lecture")
    # test_state_action_pair_cond()

    find_optimal_policy("lecture")
    find_optimal_policy("a4")
# ---------------------------------------------------------------------------------

def dump_world_info(world: str):
    print(f"discount factor of {world}:", rw.get_gamma(world))
    print(
        f"immediate reward of non-goal state of {world}:", rw.get_reward(world))
    print(f"grid for {world}:\n", rw.read_grid(world))
    # make_move(grid, curr_state, dir_intended, world)


def provided_helpers(world: str):
    grid = rw.read_grid(world)
    print("next states:", rw.get_next_states(grid, (0, 0)))
    print("is_goal:", rw.is_goal(grid, (0, 0)))
    print(rw.get_next_states(grid, (1, 1)))

def test_get_reward_by_state(world):
    print(rw.read_grid(world))
    print(get_reward_by_state(world, rw.read_grid(world), (0, 0)))
    print(get_reward_by_state(world, rw.read_grid(world), (0, 1)))
    print(get_reward_by_state(world, rw.read_grid(world), (1, 1)))
    print(get_reward_by_state(world, rw.read_grid(world), (1, 3)))
    print(get_reward_by_state(world, rw.read_grid(world), (2, 3)))


def test_state_action_pair_cond():
    n_sa = [[[30, 30, 30, 30],
             [24, 21, 14, 16],
             [15, 16, 10, 11],
             [12, 8, 10, 10]],

            [[23, 30, 22, 30],
             [0,  0,  0,  0],
             [5,  3,  3,  7],
             [3,  3,  0,  3]],

            [[17, 11, 17, 11],
             [8,  9, 11, 12],
             [5,  1,  0,  5],
             [2,  0,  0,  0]]]

    n_sa2 = [[[330, 330, 330, 330],
             [324, 321, 314, 316],
             [315, 316, 310, 311],
             [312, 338, 130, 310]],

            [[330, 330, 330, 330],
             [324, 321, 314, 316],
             [315, 316, 310, 311],
             [312, 338, 130, 310]],

            [[330, 330, 330, 330],
             [324, 321, 314, 316],
             [315, 316, 310, 311],
             [312, 338, 130, 310]]]
             
    print(state_action_pair_cond(rw.read_grid("lecture"), n_sa))
    print(state_action_pair_cond(rw.read_grid("lecture"), n_sa2))

if __name__ == "__main__":
    main()
