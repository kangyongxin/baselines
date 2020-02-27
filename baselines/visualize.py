
import numpy as np
import matplotlib.pyplot as plt 
filepath = 'episode_rewardsmy'
filename= filepath + '.npy'
a=np.load(filename)
a=a.tolist()
plt.plot(a)
plt.savefig(filepath+'.png')