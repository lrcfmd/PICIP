import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

'''
# Define the points: start, stop, sample
start = np.array([x1, y1])  # Replace x1, y1 with actual coordinates
stop = np.array([x2, y2])   # Replace x2, y2 with actual coordinates
sample = np.array([sx, sy])  # Replace sx, sy with actual coordinates

# 1) Calculate vectors from the sample to start and stop
v1 = start - sample
v2 = stop - sample

# 2) Compute the angle between the two vectors
cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians

# 3) Given n_l, calculate delta_angle
n_l = 10  # For example, let's say we want 10 intervals
delta_angle = angle / (n_l - 1)  # Dividing the angle into n_l parts

# 4) Create a rotation matrix function
def rotate_vector(v, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    return np.dot(rotation_matrix, v)

# 5) Generate points by rotating v1 by increments of delta_angle
rotated_points = [start]  # Start with the initial point
for i in range(1, n_l):
    # Rotate the v1 vector by i * delta_angle
    rotated_vector = rotate_vector(v1, i * delta_angle)
    rotated_points.append(sample + rotated_vector)

rotated_points = np.array(rotated_points)

# 6) Visualize the result

plt.figure(figsize=(6, 6))

# Plot the start, stop, and sample points
plt.scatter(*start, color='green', label='Start')
plt.scatter(*stop, color='blue', label='Stop')
plt.scatter(*sample, color='red', label='Sample')

# Plot lines from sample to each rotated point
for point in rotated_points:
    plt.plot([sample[0], point[0]], [sample[1], point[1]], 'b-')
    plt.scatter(*point, color='orange')  # Mark the rotated point

# Adjust the plot
plt.xlim([min([start[0], stop[0], sample[0]]) - 1, max([start[0], stop[0], sample[0]]) + 1])
plt.ylim([min([start[1], stop[1], sample[1]]) - 1, max([start[1], stop[1], sample[1]]) + 1])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend()
plt.title(f"Lines Emanating from Sample Point with Equal Delta Angle ({np.degrees(delta_angle):.2f}°)")
plt.show()
'''
a=2
b=2
x=[0.0, 0.10987993670713558, 0.19431603650799814, 0.2612979201276631, 0.31578744387788765, 0.3610288759642449, 0.3992329337483844, 0.4319580338774021, 0.4603343779407765, 0.4852016010668222, 0.5071965323251345, 0.5268109599138794, 0.5444307171524467, 0.5603627758425579, 0.5748544311287573, 0.5881071461772627, 0.6002867140315772, 0.611530831109484, 0.6219548202280452, 0.6316560100463929, 0.640717125109163, 0.6492089378439001, 0.657192363446477, 0.6647201296133006, 0.6718381185296899, 0.6785864538285576, 0.6850003873655294, 0.6911110275811808, 0.6969459415513142, 0.7025296556041584, 0.7078840739395807, 0.7130288305462428, 0.7179815865398229, 0.7227582825946022, 0.7273733542339151, 0.7318399162512271, 0.7361699213559302, 0.740374297203748, 0.7444630652262946, 0.7484454440764139, 0.7523299400237609, 0.756124426244381, 0.759836212629959, 0.7634721074823452, 0.7670384722454439, 0.7705412702506031, 0.7739861103061263, 0.7773782858408413, 0.7807228102112459, 0.7840244486980583, 0.7872877476480953, 0.7905170611589759, 0.7937165756552972, 0.7968903326641451, 0.8000422500638297, 0.8031761420515939, 0.8062957380529011, 0.8094047007761425, 0.8125066436016836, 0.8156051474826916, 0.8187037775268577, 0.8218060994227236, 0.8249156958716969, 0.8280361831869408, 0.8311712282231347, 0.8343245658067098, 0.8375000168447121, 0.8407015073021484, 0.8439330882528534, 0.8471989572279718, 0.8505034811096269, 0.8538512208458652, 0.8572469582973923, 0.8606957255679283, 0.8642028372194777, 0.8677739258329904, 0.8714149814456819, 0.875132395481061, 0.8789330098894237, 0.8828241723388761, 0.8868137984444014, 0.8909104422008568, 0.895123376002267, 0.8994626818935624, 0.9039393560236006, 0.9085654286648251, 0.9133541026543956, 0.918319913718857, 0.9234789169016242, 0.9288489042620364, 0.9344496602121566, 0.940303262376953, 0.9464344378040528, 0.9528709868447056, 0.9596442902598975, 0.9667899193246343, 0.9743483742557107, 0.9823659836588371, 0.9908960075663706, 1.0]

beta_pdf = beta.pdf(x, a, b)
print(beta_pdf)

