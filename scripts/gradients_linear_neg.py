import numpy as np
import matplotlib.pyplot as plt

def forward_backward_prop(w,T):
    hs = [0.5]
    for _ in range(T):
        hs.append(w * hs[-1])

    dh = 1
    for _ in range(T):
        dh = w * dh
        
    return hs[-1], dh
    
T = 11     #sequence length
wlim = 4   #limit of interval over weights w

results = []
ws = np.linspace(-wlim, wlim, 1000)
for w in ws:
    results.append(forward_backward_prop(w, T))
    
plt.plot(ws, [r[0] for r in results], label='RNN state', color='darkblue', linewidth=3.0)
plt.plot(ws, [r[1] for r in results], label='Gradient', color='red', linewidth=3.0)
plt.title('Exploding gradients on network weights subject to linear activation')
plt.xlabel('Value of weight w')
plt.ylabel('Value of state or gradient')
plt.legend()
plt.savefig('exploding_gradient_linear_neg.png', dpi=300)