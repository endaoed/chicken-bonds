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
    scatt_1 = fig.add_scatter(name="summed beta")
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
    scatt_1 = fig.add_scatter(name="composite Beta")
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


r1, r2, r3 = np.array([0, 35]), np.array([1e3, 5e6]), np.array([1, 10])
v1, v2 = [12, 3.2e3, 3], [35, 1.78e4, 2e153]

ranges = [[r1, "polynomial", 3], [r2, "log", 2], [r3, "uniform", 1]]
versions = [v1, v2]


def modifier(style:str, velocity=None):
    assert style in ["log", "polynomial", "uniform"] # add as many as you want
    if style == "polynomial":
        assert velocity != None
        return lambda x : x ** velocity
    elif style == "log":
        assert velocity != None
        g = lambda x : np.log(x) / np.log(2)
        return lambda x : g(x) + abs(g(x)[np.isfinite(g(x))].min())
    elif style == "uniform":
        return lambda x : "uniform"

def range_to_unit(r, val= None):
    """r is an np array of the range interval"""
    if val == "uniform":
        return "uniform"
    r = np.array(r)
    r_max, r_min = r.max(), r.min()
    if val == None:
        return (r - r_min)/(r_max - r_min)
    else:
        return (val - r_min)/(r_max - r_min)

def unit_to_alpha_beta(val):
    """Converty modified range to allowed alpha range 1.5 - 10.5"""
    if val == "uniform": 
        return 1, 1
    a_min, a_max = 1.5, 10.5
#     alpha_range = np.array([a_min, a_max])
    alpha = val * (a_max - a_min) + a_min
    return alpha, 12 - alpha

def compose(instant_values):
    """compose params across vars"""
    alpha, beta = instant_values.sum(axis=0)
    return alpha, beta

def instance(versions, ranges, verbose=False):
    """identify an actual value"""
    version_param_mat = []
    for v in versions:
        individual_parameter_list = []
        for i, range_details in enumerate(ranges):
            r, mod, vel = range_details
            f = modifier(mod, vel)
            modified = f(v[i])
            unit_modified = range_to_unit(f(r), val=modified) # scale everything ofc
#             print("version" , v[i])
#             print("Modified unit: ", unit_modified)
            alpha, beta = unit_to_alpha_beta(unit_modified)
            individual_parameter_list.append([alpha, beta])
        individual_parameter_list = np.array(individual_parameter_list)
        composed_parameters = compose(individual_parameter_list)
        version_param_mat.append([individual_parameter_list, composed_parameters])
        if verbose:
            print("alpha, beta matrix: ", individual_parameter_list)
            print("composed parameters: ", composed_parameters)
    return version_param_mat

def plot_summary():
    _, ax = plt.subplots(2, 2, figsize=(12, 8))
    for i, v in enumerate(instance(versions, ranges)):
        param_list, param_sum = v
        x = np.linspace(0, 1, 100)
        b1 = beta(*param_list[0]).pdf(x)
        b2 = beta(*param_list[1]).pdf(x)
        b3 = beta(*param_list[2]).pdf(x)
#         print(i, param_list, "\n\n")
        ax[0, i].set(title="Individual Beta Distributions V{}".format(i+1))
        ax[0, i].plot(x, b1, x, b2, x, b3)
        ax[1, i].set(title="Composed Beta Distributions V{}".format(i+1))
        ax[1, i].plot(x, beta(*param_sum).pdf(x))

