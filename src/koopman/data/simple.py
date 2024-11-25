from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_fun(mu, lamb):

    def fun(t, x):

        x_dot = np.zeros_like(x)
        x_dot[0] = mu*x[0]
        x_dot[1] = lamb*(x[1]-x[0]**2)
        return x_dot
    
    return fun


def proof_of_concept_simple():
    mu = -0.05
    lamb = -1.0

    fun = get_fun(mu, lamb)
    t_span = (0, 100)
    for x in np.linspace(-1,1,10):
        for y in [-5, 5]:
            x0 = np.array([x, y])
            sol = solve_ivp(fun, t_span, x0, t_eval=np.linspace(*t_span, 100), vectorized=True)

            colors = cm.jet(sol.t / sol.t.max())
            plt.scatter(sol.y[0], sol.y[1], c=colors)
            plt.plot(sol.y[0], sol.y[1])
    plt.show()

def generate_simple(time_stop, n_points, mu, lamb):

    fun = get_fun(mu, lamb)
    t_span = (0, time_stop)
    x1 = np.random.uniform(-0.5, 0.5)
    x2 = np.random.uniform(-0.5, 0.5)
    x_init = tuple([x1, x2])
    sol = solve_ivp(fun, t_span, x_init, t_eval=np.linspace(*t_span, n_points), vectorized=True)
    return x_init, sol.y.T[None, :, :]