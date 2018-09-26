import numpy as np

states = ('Healthy', 'Fever')

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}

emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}
class HMM:
    """
    Order 1 Hidden Markov Model

    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi


def generate_index_map(lables):
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label


states_label_index, states_index_label = generate_index_map(states)
observations_label_index, observations_index_label = generate_index_map(observations)


def convert_observations_to_index(observations, label_index):
    list = []
    for o in observations:
        list.append(label_index[o])
    return list


def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v


def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m


A = convert_map_to_matrix(transition_probability, states_label_index, states_label_index)
print (A)
B = convert_map_to_matrix(emission_probability, states_label_index, observations_label_index)
print(B)
observations_index = convert_observations_to_index(observations, observations_label_index)
pi = convert_map_to_vector(start_probability, states_label_index)
print(pi)

hmm = HMM(A, B, pi)



def simulate(self, T):
    def draw_from(probs):
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(self.pi)
    observations[0] = draw_from(self.B[states[0], :])
    for t in range(1, T):
        states[t] = draw_from(self.A[states[t - 1], :])
        observations[t] = draw_from(self.B[states[t], :])
    return observations, states