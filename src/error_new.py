import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from algorithm import *

# Function to plot Beta distribution
def plot_beta_distribution(a, b):
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, a, b)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"Beta Distribution (a={a}, b={b})")
    plt.fill_between(x, y, alpha=0.2)
    plt.title(f"Beta Distribution for a={a}, b={b}")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_beta_parameters(mean, sigma):
    # Mean is µ, sigma is the standard deviation
    if mean <= 0 or mean >= 1:
        raise ValueError("Mean must be between 0 and 1.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    
    # Calculate a and b using the mean and sigma
    variance = sigma ** 2
    common_factor = mean * (1 - mean) / variance - 1
    a = mean * common_factor
    b = (1 - mean) * common_factor
    
    return a, b


'''
means=np.linspace(0.1,0.9,10)
stds=[0.02,0.05,0.1]

for mean in means:
    for std in stds:
        a, b = calculate_beta_parameters(mean, std)
        plot_beta_distribution(a, b)
        '''

model=all_information()
basis=np.array([[-0.7171759 ,  0.48331419, -0.21827093,  0.45213263],
       [ 0.32141217, -0.38569461, -0.57854191,  0.64282435]])
contained_point=[0,2,3,0]
species_dict={'Mg':2,'B':3,'O':-2,'F':-1}
model.setup_from_species_dict(species_dict,contained_point=contained_point,basis=basis,cube_size=200)


knowns=['MgO','MgF2']
k_weights=np.array([0.5,0.5])
u=['Mg3B(OF)3']
s_weights=[0.5,0.5,0.5]
s_weights=np.array(s_weights)/sum(s_weights)
allc=knowns+u
allc=[Composition(x) for x in allc]
all_c_s=[[s.get_atomic_fraction(el) for el in model.phase_field] for s in allc]
s=np.average(all_c_s,axis=0,weights=s_weights)
s=model.convert_point_to_constrained(s)
model.add_sample(knowns,k_weights,s,std=0.02,n_l=100,n_p=30)
p=model.values

model.find_corners_edges(model.normal_vectors)
plot=Square(model)
plot.make_binaries_as_corners()
plot.create_fig()
plot.plot_p(model.omega,p,s=2)
print(len(p),len(model.omega))
#plot(len(p.create_fig_corners(model.corners,model.edges)
plot.show(s=10)

