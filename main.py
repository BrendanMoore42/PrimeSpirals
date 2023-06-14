"""
Prime Spiral Maker
Generates an infinite number of spirals based on any given series in 2d or 3d.

author: brendanmoore42@github.com
date: June 2023
"""
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class SpiralMaker:
    def __init__(self, title, output_dir, low, high, series="primes", plot_type="scatter", random_colours=True, save_figure=False, degrees=360, modifier=None, deg_modifier=None, iterations=1000):
        # Set directories
        self.title = title + "-" + plot_type  # title of current task plus plot type, ex. "Primes_2_100"
        self.output_dir = output_dir
        self.plot_type = plot_type  # Can be scatter, scatter3d, scatter4d

        # Set integer sequence
        if series == "primes":
            self.low = low  # lower bound for primes
            self.high = high  # upper bound for primes
            self.series = self.generate_primes(low, high)
        else:  # if there is a sequence other than primes to try, this takes a list
            self.series = series  # list of integers

        # Spiral variables
        self.degrees = degrees
        self.iterations = iterations
        if not modifier:
            self.modifier = 1
        else:
            self.modifier = modifier
        self.deg_modifier = deg_modifier

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
                            '#f6abb6', '#d41752', '#3d1e6d']

    def generate_primes(self, low, high):
        primes = []
        for i in range(low, high):
            for j in range(2, int(i / 2) + 1):
                if i % j == 0:
                    break
            else:
                primes.append(i)

        return primes

    def generate_xy(self, theta=None, degrees=360, deg_modifier=5, cycles=1000):
        """Create a list of xy points"""
        # Set theta or make a new one
        theta = theta if theta else np.radians(np.linspace(0, degrees*deg_modifier, cycles))
        r = theta**2
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        return x, y

    def plot_spiral(self, x, y, mod_variable, plot_size=(8, 8), plot_type="plot", title="", hide_ax=False):
        """Main function: Create directories, generate spirals"""
        temp_dir, num = title.split("_")[0], title.split("_")[1]
        sizes = []  # list for increasing plot size as spirals enlarge
        size_modifier = 1
        colours = []
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

            # Generate list of colours to x, y
            colours.append(self.colours[int((int(num)*i*self.modifier)) % len(self.colours)])
            # Set up x, y lists
            temp_x.append(int(xy[0]))
            temp_y.append(int(xy[1]))

            # Set z coordinates for 3d space
            if plot_type == 'scatter3d':
                z.append(i/100)  # change val here to get different z heights

        if plot_type == 'scatter3d':
            fig = plt.figure(figsize=plot_size)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, z, c=colours, s=sizes, alpha=self.alpha)

        else:  # normal 2d plot
            plt.figure(figsize=plot_size)
            if not hide_ax:
                plt.axhline(c='black', alpha=0.2)
                plt.axvline(c='black', alpha=0.2)
            if plot_type == 'plot':
                plt.plot(x, y)
            if plot_type == 'scatter':
                plt.scatter(x, y, s=sizes, c=colours, alpha=self.alpha)

            try:
                sns.despine()
            except:
                pass

        if plot_type == 'scatter':
            plt.axis('off')
            plt.title(num, y=-0.01, loc='right', c='#686868', fontsize=25)
        else:
            plt.title(num, y=1, pad=-35, loc='right', fontsize=25)

        if self.save_figure:
            if os.path.isdir(f'{self.output_dir}/{self.title}'):
                pass
            else:
                os.mkdir(f'{self.output_dir}/{self.title}')
            plt.savefig(f'{self.output_dir}/{self.title}/{num}', dpi=400, transparent=True)
        else:
            plt.show()
        plt.close()

    def run_spiral_maker(self):
        """Main function that puts it all together, outputs images to specified directory"""
        # Run analysis, skip spirals existing directories and spirals
        for p in self.series:
            if os.path.isfile(f"{self.output_dir}/{self.title}/{str(p)}.png") and self.save_figure:
                continue
            else:
                x, y = self.generate_xy(None, self.degrees*int(p), self.modifier*p, self.iterations)
                self.plot_spiral(x, y, self.modifier*p, plot_type=self.plot_type, title=self.title+'_'+str(p), hide_ax=True)

        # Write parameters of analysis to output dir
        params = f"Plot: {self.plot_type}, Lower bound: {str(self.low)}, Upper bound: {str(self.high)}, " \
                 f"Degrees: {str(self.degrees)}, Degree Modifier: {round(self.deg_modifier, 3)}, " \
                 f"Modifier: {round(self.modifier, 3)}, Iterations: {str(self.iterations)}"

        with open(f"{self.output_dir}/{self.title}/params.txt", 'w') as f:
            f.write(params)

    def visualize_palette(self, colours):
        """Create simply line chart to display colours"""
        x = np.linspace(0, 1, 10)
        fig, ax = plt.subplots()
        for i in range(0, len(colours)):
            plt.plot(x, i * x + i, color=colours[i])
        plt.show()


# Set modifiers
phi = (1+np.sqrt(5)) / 2  # phi is a good modifier
lower_bound, upper_bound = 100, 150  # lowest bound == 2, upper is anything the computer can handle
plot_type = 'scatter3d'  # scatter for 2d plot, scatter3d for 3d plot

# Ensure each title is separated by '-' as the image number is separated by '_'
sp = SpiralMaker(title=f"Primes-{lower_bound}-{upper_bound}", output_dir="output",
                 low=lower_bound, high=upper_bound, random_colours=False,
                 deg_modifier=phi*2, modifier=np.pi*6,  # play with modifiers to generate different results
                 plot_type=plot_type, save_figure=True)  # Primes start at low bound and up to higher bound
sp.run_spiral_maker()
