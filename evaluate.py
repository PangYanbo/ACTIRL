import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


def eva(qlr,irlr):
def main():
    demo = []
    sim = []

    state_idx = {'0': 'home', '1': 'work', '2': 'school', '3': 'other', '4': 'train', '5': 'vehicle', '6': 'walk'}

    df_truth = pd.DataFrame({'0': np.zeros(6),
                             '1': np.zeros(6),
                             '2': np.zeros(6),
                             '3': np.zeros(6),
                             '4': np.zeros(6),
                             '5': np.zeros(6),
                             '6': np.zeros(6)
    })

    df_sim = pd.DataFrame({'0': np.zeros(6),
                       '1': np.zeros(6),
                       '2': np.zeros(6),
                       '3': np.zeros(6),
                       '4': np.zeros(6),
                       '5': np.zeros(6),
                       '6': np.zeros(6)
    })

    with open('test.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            demo.append(line)

    with open(str(qlr)+'_'+str(irlr)+'out.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            sim.append(line)

    for t in range(6):
        for i in range(len(demo)):
            df_truth[demo[i][t]][t] += 1
            df_sim[sim[i][t]][t] += 1

    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.title('Results of activity ratio at each time step')
    for i in range(7):
        plt.subplot(3, 3, i+1)
        plt.plot(range(6), df_truth[str(i)]/len(demo), label='truth')
        plt.plot(range(6), df_sim[str(i)]/len(demo), label='sim')
        plt.ylim(0, 1)
        plt.legend()
        plt.title(state_idx[str(i)])

    plt.savefig('eva_'+str(qlr)+'_'+str(irlr)+'.png')


def main():
    param = [(qlr, irlr) for qlr in np.arange(0.01, 0.1, 0.01) for irlr in np.arange(0.01, 0.1, 0.01)]

    for p in param:
        eva(p[0], p[1])


if __name__ == '__main__':
    main()
