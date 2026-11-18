import numpy as np

def write_W_b(nn):
    W_b = nn.serveForVerify()
    for i in range(int(len(W_b)/2)):
        filename1 = "W" + str(i + 1) + ".csv"
        filename2 = "b" + str(i + 1) + ".csv"
        np.savetxt(filename1, W_b[i * 2])
        np.savetxt(filename2, W_b[i * 2 + 1].tolist())


if __name__ == '__main__':
    area = [ [ -2, 2 ], [ -2, 2 ] ]
    area = np.array(area)
    low = area[:, 0]  # 每维下界
    high = area[:, 1]
    point = np.random.uniform(low, high)
    print(point)