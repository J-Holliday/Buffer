# 1. Preprocess
import random
# 1.1. Make NeuralLetwork
# 1.1.1. Define Layers
n_hidden = 2
n_layer = n_hidden + 2

# 1.1.2. Define Units
n_unit_i = 3
n_unit_h = 5 # all 
n_unit_o = 2

unit_i = [0 for u in range(n_unit_i)]
unit_h1 = [0 for u in range(n_unit_h)]
unit_h2 = [0 for u in range(n_unit_h)]
unit_o = [0 for u in range(n_unit_o)]

# 1.1.3. Initialize weight
w1 = [[random.uniform(-1, 1) for u_before in range(n_unit_i)] for u_after in range(n_unit_h)]
w2 = [[random.uniform(-1, 1) for u_before in range(n_unit_h)] for u_after in range(n_unit_h)]
w3 = [[random.uniform(-1, 1) for u_before in range(n_unit_h)] for u_after in range(n_unit_o)]

# 1.2. Define parameter
n_learn = 50000
alpha = 0.1 # learn Rate
train = [[0, 0], [0, 1], [1, 0], [1, 1]]
for t in train:
    t.insert(0, 1)
teach = [[0, 1], [1, 0], [1, 0], [0, 1]]

# 1.3 Implement forward propagation
import math

# 1.3.1 Define activation fucntion
def sigmoid(z):
    if z > 10: return 0.99999
    elif z < -10: return 0.00001
    else: return 1 / (1 + math.exp(-1 * z))
    
# 1.3.2 forward propagation

def forward(train_vec):
    
    for i in range(n_unit_i):
        unit_i[i] = train_vec[i]
    unit_i[0] = 1
        
    # 1.3.2.1 forward between input-hidden1
    for h1 in range(n_unit_h):
        buf = 0
        for i in range(n_unit_i):
            buf += unit_i[i] * w1[h1][i]
        unit_h1[h1] = sigmoid(buf)
    unit_h1[0] = 1
        
    # 1.3.2.2 forward between hidden1-hidden2
    for h2 in range(n_unit_h):
        buf = 0
        for h1 in range(n_unit_h):
            buf += unit_h1[h1] * w2[h2][h1]
        unit_h2[h2] = sigmoid(buf)
    unit_h2[0] = 1

    # 1.3.2.3 forward between hidden2-output
    for o in range(n_unit_o):
        buf = 0
        for h2 in range(n_unit_h):
            buf += unit_h2[h2] * w3[o][h2]
        unit_o[o] = sigmoid(buf)

# 1.3.3 back propagation

def backpropagation(teach_vec):
    
    # 1.3.3.1 get cost
    buf = 0
    for o in range(n_unit_o):
        buf += (teach_vec[o] - unit_o[o]) ** 2
    cost = buf / 2
    
    # 1.3.3.2 get grad between hidden2-output
    for o in range(n_unit_o):
        for h2 in range(n_unit_h):
            delta = (unit_o[o] - teach_vec[o]) * unit_o[o] * (1 - unit_o[o]) * unit_h2[h2]
            w3[o][h2] -= alpha * delta
            
    # 1.3.3.3 get grad
    for o in range(n_unit_o):
        for h2 in range(n_unit_h):
            for h1 in range(n_unit_h):
                delta = ((unit_o[o] - teach_vec[o]) * unit_o[o] * (1 - unit_o[o])
                         * w3[o][h2] * unit_h2[h2] * (1 - unit_h2[h2]) * unit_h1[h1])
                w2[h2][h1] -= alpha * delta
                
    # 1.3.3.4 get grad
    for o in range(n_unit_o):
        for h2 in range(n_unit_h):
            for h1 in range(n_unit_h):
                for i in range(n_unit_i):
                    delta = ((unit_o[o] - teach_vec[o]) * unit_o[o] * (1 - unit_o[o])
                             * w3[o][h2] * unit_h2[h2] * (1 - unit_h2[h2])
                             * w2[h2][h1] * unit_h1[h1] * (1 -unit_h1[h1]) * unit_i[i])
                    w1[h1][i] -= alpha *delta
                    
    return cost
    
import matplotlib.pyplot as plt
#%matplotlib inline
plt_x = []
plt_y = []

# for ppm
import subprocess
import PPM
ppm = PPM.PPM()
buf_c = 0
n_ppm = 0

for n in range(n_learn):
    forward(train[n%4])
    c = backpropagation(teach[n%4])
    plt_x.append(n)
    plt_y.append(c)
    
    # for ppm
    buf_c = c - buf_c
    if ppm.buffering2((c, buf_c), n):
        ppm.executor(ppm.new_image(), "output/output_" + str(n_ppm))
        n_ppm += 1
        if n_ppm%20 == 0:
            try:
                popen.terminate()
                #print(buf_c)
                print("n = " + str(n))
            except:
                pass
            finally:
                popen = subprocess.Popen(["sudo", "./led-image-viewer", "output/output_{n}.ppm".format(n=n_ppm), "-r", "16"])
        
print("ppm.error_means-----")
print(ppm.error_means)
print("ppm.grad_means-----")
print(ppm.grad_means)
#plt.plot(plt_x, plt_y)
#plt.xlim(0, 50000)
#plt.ylim(0, 0.5)
#plt.show()
