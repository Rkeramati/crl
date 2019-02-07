import time
from cvar.gridworld.cliffwalker import *


def epoch(world, policy, max_iters=100, plot_machine=None):
    """
    Evaluates a single epoch starting at start_state, using a given policy.
    :param start_state:
    :param policy: Policy instance
    :param max_iters: end the epoch after this #steps
    :return: States, Actions, Rewards
    """
    s = world.initial_state
    S = [s]
    A = []
    R = []
    i = 0
    t = Transition(s, 0, 0)
    while s not in world.goal_states and i < max_iters:
        a = policy.next_action(t)
        A.append(a)

        if plot_machine is not None:
            plot_machine.step(s, a)
            time.sleep(0.5)

        t = world.sample_transition(s, a)

        r = t.reward
        s = t.state

        R.append(r)
        S.append(s)
        i += 1

    return S, A, R


def optimal_path(world, policy, max_=False):
    """ Optimal deterministic path. """
    s = world.initial_state
    ss = s
    max_iteration = 20
    it = 0
    states = [s]
    t = Transition(s, 0, 0)
    t_fake = Transition(s, 0, 0)
    observations = []
    while s not in world.goal_states and it < max_iteration:
        a = policy.next_action(t_fake)
        if max_:
            temp = max(world.transitions(ss)[a], key=lambda t: t.prob)#max(np.arange(len(world.transitions(ss)[a])))#, p=[x.prob for x in world.transitions(ss)[a]])
            #t = world.transitions(ss)[a][temp]
            t = temp

            temp = max(world.transition_rmax(s, a), key=lambda t: t.prob)#max(np.arange(len(world.transition_rmax(s, a))))#, p=[x.prob for x in world.transition_rmax(s, a)])
            #t_fake = world.transition_rmax(s, a)[temp]
            t_fake = temp
        else:
            temp = np.random.choice(np.arange(len(world.transitions(ss)[a])), p=[x.prob for x in world.transitions(ss)[a]])
            t = world.transitions(ss)[a][temp]

            temp = np.random.choice(np.arange(len(world.transition_rmax(s, a))), p=[x.prob for x in world.transition_rmax(s, a)])
            t_fake = world.transition_rmax(s, a)[temp]

        ns, r = t.state, t.reward
        states.append(ns)
        observations.append((ss, a, ns, r))
        s = t_fake.state
        ss = ns
        it+=1
    return states, observations