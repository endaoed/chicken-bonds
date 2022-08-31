import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
from ipywidgets import interact
import matplotlib.pyplot as plt

def standard_beta():
    fig = go.FigureWidget()
    scatt = fig.add_scatter()

    xs=np.linspace(0, 1, 100)

    @interact(a=(0, 40.0, 0.001), b=(0, 40.0, 0.001))
    def update(a=6, b=6):
        with fig.batch_update():
            scatt.data[0].x=xs
            scatt.data[0].y=beta(a, b).pdf(xs)
    return fig

def constrained_beta():
    fig = go.FigureWidget()
    scatt = fig.add_scatter()
    xs=np.linspace(0, 1, 100)

    @interact(a=(1.5, 10.5, 0.001))
    def update(a=6):
        with fig.batch_update():
            scatt.data[0].x=xs
            scatt.data[0].y=beta(a, 12-a).pdf(xs)
    return fig

def constrained_beta_with_nonlinear_param():
    fig = go.FigureWidget()
    scatt = fig.add_scatter()
    xs=np.linspace(0, 1, 100)

    @interact(p=(0, 1, 0.001))
    def update(p=0.0):
        with fig.batch_update():
            scatt.data[0].x=xs
            scatt.data[0].y=beta(9 * p**8 + 1.5, 10.5 - 9 * p**8).pdf(xs)
    return fig


def additive_beta_composition():
    fig = go.FigureWidget()
    scatt_1 = fig.add_scatter(name="summed beta", line={"color" : "blue", "width": 4})
    scatt_2 = fig.add_scatter(name="added beta")
    xs=np.linspace(0, 1, 100)
            
    @interact(a=(1.5, 10.5, 0.001))
    def update(a=1.5):
        with fig.batch_update():
            scatt_1.data[0].x=xs
            scatt_1.data[0].y=beta(2, 20).pdf(xs) + beta(a, 12-a).pdf(xs)
            scatt_2.data[1].x=xs
            scatt_2.data[1].y=beta(a, 12-a).pdf(xs)
    trace = go.Scatter(
        x = xs,
        y = beta(2, 20).pdf(xs),
        name = "fixed beta(2, 20)",
        line = dict(
            color = "black",
            width = 4,
            dash = "dot"
        )
      )
    fig.add_trace(trace)
    
    return fig


def multiplicative_beta_composition():
    fig = go.FigureWidget()
    scatt_1 = fig.add_scatter(name="composite Beta", line={"color" : "blue", "width": 3})
    scatt_2 = fig.add_scatter(name="imposed Beta")
    xs=np.linspace(0, 1, 100)
            
    @interact(red=(1.5, 10, 0.001))
    def update(red=1.5):
        with fig.batch_update():
            scatt_1.data[0].x=xs
            scatt_1.data[0].y=beta(2 + red, 20 + 12-red).pdf(xs)
            scatt_2.data[1].x=xs
            scatt_2.data[1].y=beta(red, 12-red).pdf(xs)
#     @interact()
    trace = go.Scatter(
        x = xs,
        y = beta(2, 20).pdf(xs),
        name = "fixed beta(2, 20)",
        line = dict(
            color = "black",
            width = 4,
            dash = "dot"
        )
      )
    fig.add_trace(trace)
    return fig


def multiplicative_beta_composition2():
    fig = go.FigureWidget()
    scatt_1 = fig.add_scatter(name="composite Beta", line={"color" : "blue", "width": 3})
    scatt_2 = fig.add_scatter(name="imposed Beta 1")
    scatt_3 = fig.add_scatter(name="imposed Beta 2")
    scatt_4 = fig.add_scatter(name="imposed Beta 3")
    xs=np.linspace(0, 1, 100)
    
    slider = (1.5, 10, 0.001)
    @interact(red=slider, green=slider, purple=slider)
    def update(red=1.5, green=1.5, purple=1.5):
        with fig.batch_update():
            scatt_1.data[0].x=xs
            scatt_1.data[0].y=beta(red + green + purple, 
                                   12-red + 12-green + 12-purple).pdf(xs)
            scatt_2.data[1].x=xs
            scatt_2.data[1].y=beta(red, 12-red).pdf(xs)
            scatt_3.data[2].x=xs
            scatt_3.data[2].y=beta(green, 12-green).pdf(xs)
            scatt_4.data[3].x=xs
            scatt_4.data[3].y=beta(purple, 12-purple).pdf(xs)

    return fig



# +
def remapper(r1: np.ndarray, r2: np.ndarray, val=None):
    """general remapper. Maps r1 to r2"""
    assert type(r1) == type(r2) == np.ndarray
    min1, max1 = r1.min(), r1.max()
    min2, max2 = r2.min(), r2.max()
    if val == None:
        return (r1 - min1) * (max2 - min2) / (max1 - min1) + min2
    else:
        return (val - min1) * (max2 - min2) / (max1 - min1) + min2
    
def modifier_dict(style:str, velocity=None):
    """
    rudimentary dictionary of modifiers
    """
    assert style in ["logarithmic", "polynomial", "uniform", "linear"] # add whatever
    if style == "uniform":
        return lambda x : "uniform"
    else:
        assert velocity != None
        if style == "polynomial":
            return lambda x : x ** velocity
        elif style == "logarithmic":
            g = lambda x : np.log(x)
            return lambda x : g(x) + abs(g(x)[np.isfinite(g(x))].min())
        elif style == "linear":
            return lambda x : velocity * x

def f1_range_to_unit(var_range : np.ndarray, velocity: float, modifier: str, val=None):
    """
    Maps the variable range to the unit via modifier.
    If val=None, it outputs the remapped range
    else it behaves as a function for the val
    """
    if modifier == "uniform": return "uniform"
    unit = np.array([0, 1])
    modifier = modifier_dict(modifier, velocity)
    m = modifier(var_range)
    m_val = modifier(val)
    if val != None:
        # piecemeal logic
        if val < var_range.min():
            return 0.
        if val > var_range.max():
            return 1.
        return remapper(m, unit, m_val)
    else:
        return remapper(m, unit)

def f2_unit_to_alpha_beta(x):
    if x == "uniform":
        return 1, 1
    alpha = 9 * x + 1.5
    return alpha, 12 - alpha

def composer(propensities, values):
    """Stitch everything together"""
    param_pairs = {}
    sum_a, sum_b = 0, 0
    for var, vals in propensities.items():
        var_range, velocity, modifier = vals.values()
        unit_out = f1_range_to_unit(var_range, velocity, modifier, values[var])
        a, b = f2_unit_to_alpha_beta(unit_out)
        sum_a += a
        sum_b += b
        param_pairs[var] = {"alpha" : a, "beta" : b}
    final_params = [sum_a, sum_b]
    return param_pairs, final_params

def plot_output(output_params):
    _, ax = plt.subplots(2, 2, figsize=(12, 6))
    _.suptitle("Example Outputs")
    for i, version in enumerate(output_params):
        var_params, summed = version[0], version[1]
        beta_details = [(name, params.values()) for name, params in var_params.items()]
        x = np.linspace(0, 1, 100)
        for n, p in beta_details:
            size = 3 if n == "bond_age " else 1
            ax[0, i].plot(x, beta(*p).pdf(x), label=n)
        ax[1, i].plot(x, beta(*summed).pdf(x), label="Composite Beta")
        
    ax=ax.flatten()
    [x.legend() for x in ax]

def output_version_results(versions):
    for i, v in enumerate(versions):
        individual, composed = v
        print("\nVERSION {}\n".format(i), "\tIndividual params:")
        for n, p in individual.items():
            print("\t", n, "\t", "a: {:.2f}, b: {:.2f}".format(*p.values()))
        print("\nComposed params: \t a: {:.2f} b: {:.2f}\n".format(*composed))
    return
