import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import env
import maxent

N_STEP, N_STATES, N_ACTIONS, N_FEATURES = 6, 7, 7, 10


discount = 0.99

# q_learning_rate = 0.1
# inverse_learning_rate = 0.1

np.random.seed(1)


def update_q_table(t, state, action, reward, next_state, q_learning_rate, q_table):
    q_1 = q_table[t][int(state)][int(action)]
    q_2 = reward + discount * max(q_table[t+1][int(next_state)])
    q_table[t][int(state)][int(action)] += q_learning_rate * (q_2 - q_1)


def choose_action(values, greedy=0.2):
    if np.random.uniform() > greedy:
        c = np.max(values)
        exp_v = np.exp(values-c)
        sum_exp_v = np.sum(exp_v)
        v = exp_v / sum_exp_v
        action = str(np.random.choice(range(len(values)), p=v))
    else:
        action = str(np.random.choice(range(len(values))))

    return action


def train(q_learning_rate, inverse_learning_rate):
    e = env.Env(N_STEP, N_STATES, N_ACTIONS, N_FEATURES)
    demonstrations = np.loadtxt('test.csv', delimiter=',', dtype=str)
    e.get_init(demonstrations)
    q_table = np.random.uniform(size=(N_STEP, N_STATES, N_ACTIONS))     
    feature_expectations = np.zeros(N_FEATURES)
    maxent.find_feature_expectations(demonstrations, feature_expectations, e)

    irl_feature_expectations = np.zeros(N_FEATURES)

    alpha = np.random.uniform(size=(N_FEATURES,))
    e.set_alpha(alpha)

    grad = []

    for episode in range(2500000):
        state = e.reset()
        t = 0

        if episode != 0 and episode % 50000 == 0:
            # update alpha
            # print(episode, q_table)
            # print(episode, irl_feature_expectations)
            learner = irl_feature_expectations / float(episode)
            gradient = maxent.irl(feature_expectations, learner, alpha, inverse_learning_rate)
            print(gradient)
            grad.append(np.linalg.norm(gradient))
            e.set_alpha(alpha)

        series = [state]
        irl_feature_expectations += e.feature_vector(series)
        while True:
            action = choose_action(q_table[t][int(state)])

            next_state = e.step(action)
            series.append(next_state)

            reward = e.get_reward(series)
            update_q_table(t, state, action, reward, next_state, q_learning_rate, q_table)
            irl_feature_expectations += e.feature_vector(series)
            t += 1
            state = next_state

            if t == 5:
                break

    print(alpha)
    print(grad)
    plt.plot(grad, label='q_learning_rate: '+str(q_learning_rate)+' inverse_learning_rate: '+str(inverse_learning_rate))
    # plt.ylim(0, int(max(grad))+1)
    plt.title('q_learning_rate: '+str(q_learning_rate)+' inverse_learning_rate: '+str(inverse_learning_rate))
    plt.savefig('train_'+str(q_learning_rate)+'_'+str(inverse_learning_rate)+'.png')

    episodes = []
    for demo in demonstrations:
        episode = demo[0]
        state = demo[0]
        t = 0
        while True:
            action = choose_action(q_table[t][int(state)], greedy=0)
            next_state = e.step(action)

            t += 1
            state = next_state
            episode += state

            if t == 5:
                break
        episodes.append(episode)

    with open(str(q_learning_rate)+'_'+str(inverse_learning_rate)+'out.csv', 'w') as w:
        for episode in episodes:
            w.write(episode)
            w.write('\n')


def main():
    train()


if __name__ == '__main__':
    main()
