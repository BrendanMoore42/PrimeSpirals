"""
Prime Spiral Maker
Generate visually pleasing and neat spirals in 2 and 3D. Can be any sequence of numbers,
though I'm using Primes, because I just think they are neat.

author: brendanmoore42@github.com
date: June 2023
"""
import os
import glob
import random
import math as m
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import seaborn as sns

from collections import Counter
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch

# Visualize palette
# x = np.linspace(0, 1, 10)
# fig, ax = plt.subplots()
# for i in range(0, len(colours)):
#     plt.plot(x, i * x + i, color=colours[i])
# plt.show()

class SpiralMaker:
    def __int__(self, output_dir, random_colours=True, save_figure=False, theta=None, degrees=360, modifier=5, iterations=1000):
        # Set directories
        self.output_dir = output_dir

        # Spiral variables
        self.theta = theta
        self.degrees = degrees
        self.modifier = modifier
        self.iterations = iterations

        # Plot variables
        self.alpha = 0.8
        self.save_figure = save_figure

        # Set up colour palettes
        # Setting up colours, random sample vs. select palette
        if random_colours:
            overlap = sorted(list(name[1] for name in mcolors.CSS4_COLORS.items() if f'xkcd:{name[0]}' in mcolors.XKCD_COLORS))
            self.colours = random.sample(overlap, 9)
        else:
            self.colours = ['#ee4035', '#f37736', '#fdf498',
                            '#7bc043', '#0392cf', '#63ace5',
                            '#f6abb6', '#851e3e', '#3d1e6d']

    def generate_xy(self):
        """Create a list of xy points"""
        # Set theta or make a new one
        theta = self.theta if self.theta else np.radians(np.linspace(0, self.degrees*self.modifier, self.iterations))
        r = theta**2
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        return x, y

    def rotate_matrix(self, matrix):
        """Rotates a matrix"""
        temp_matrix = []

        for col in range(len(matrix)):
            temp_list = []
            for row in range(len(matrix)-1, -1, -1):
                temp_list.append(matrix[row][col])
            temp_matrix.append(temp_list)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                matrix[i][j] = temp_matrix[i][j]

        return matrix

    def plot_spiral(self, x, y, mod_variable, plot_size=[8, 8], plot_type="plot", title="", hide_ax=False):
        """Main function: Create directories, generate spirals"""
        temp_dir, num = title.split("_")[0], title.split('_')[1]
        sizes = []  # list for increasing plot size as spirals enlarge
        size_modifier = 1
        temp_x, temp_y, z = [], [], []
        for i, xy in enumerate(zip(x, y)):
            if plot_type == 'scatter':
                if i >= 15:
                    pass
                else:
                    size_modifier = i + (1 / (i + 1))
                sizes.append(int((i * 1 / mod_variable * int(num))) * size_modifier)
            else:
                sizes.append(int((i * 1 / mod_variable * int(num))))

            temp_x.append(int(xy[0]))
            temp_y.append(int(xy[1]))

            if plot_type in ['scatter3d', 'scatter4d']:
                if int(num) > 100:
                    i = i * 1000
                elif int(num) > 500:
                    i = i * 10000
                z.append(i*100)

        if plot_type == 'scatter3d':
            fig = plt.figure(figsize=plot_size)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, z, c=self.colours, s=sizes, alpha=self.alpha)
        if plot_type == 'scatter4d':
            fig = plt.figure(figsize=plot_size)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, z, c=self.colours, s=sizes, alpha=self.alpha)

            # Rotate matrix
            v1 = np.array([x, y, z])
            v1r = self.rotate_matrix(v1)
            v2 = np.array_split(v1r, 3)
            temp_v = []
            for i in v2:
                temp_v.append(list(i))
            x1, y1, z1 = temp_v[0], temp_v[1], temp_v[2]
            ax.scatter(list(map(lambda a: -x, x1)), y1, list(map(lambda b: -x, z1)), c=self.colours, s=sizes, alpha=self.alpha)
        else:
            plt.figure(figsize=plot_size)
            if not hide_ax:
                plt.axhline(c='black', alpha=0.2)
                plt.axvline(c='black', alpha=0.2)
            if plot_type == 'plot':
                plt.plot(x, y)
            if plot_type == 'scatter':
                plt.scatter(x, y, c=self.colours, s=sizes, alpha=self.alpha)

            try:
                sns.despine()
            except:
                pass

        if plot_type == 'scatter':
            plt.axis('off')
            plt.title(num, y=-0.01, loc='right', c='#686868', font='Rubix', fontsize=25)
        else:
            new_title = title.split("_")[2:]
            new_title = f'{new_title[2]}t-{new_title[0]}:{new_title[1]}, var:{new_title[3].split("_")[0]}, {new_title[3].split("_")[1]}'
            plt.title(new_title, y=1, pad=-35, loc='right', font='Lucida Sans Unicode', fontsize=25)

        if self.save_figure:
            if title[:2] == '2d':
                if os.path.isdir(f'{self.output_dir}/{temp_dir[2:]}/2d'):
                    pass
                else:
                    os.mkdir(f'{self.output_dir}/{temp_dir[2:]}/2d')
                plt.savefig(f'{self.output_dir}/{temp_dir[2:]}/2d/{num}', dpi=400)
            else:
                if os.path.isdir(f'{self.output_dir}/{temp_dir}'):
                    pass
                else:
                    os.mkdir(f'{self.output_dir}/{temp_dir}')
                plt.savefig(f'{self.output_dir}/{temp_dir}/{num}', dpi=400, transparent=True)
        else:
            plt.show()
        plt.close()

    def generate_primes(self, low, high):
        non_primes = set(q for w in range(2, 8) for q in range(w*2, high, w))
        primes = [x for x in range(low, high) if x not in non_primes]

        print(primes)

sp = SpiralMaker()
sp.generate_primes(1, 2000)