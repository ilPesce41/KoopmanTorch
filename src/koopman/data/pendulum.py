from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_fun():

    def fun(t, x):

        x_dot = np.zeros_like(x)
        x_dot[0] = x[1]
        x_dot[1] = -np.sin(x[0])
        return x_dot
    
    return fun

def proof_of_concept_pendulum():
    fun = get_fun()
    t_span = (0, 5)
    for x1 in range(100):
        x1 = np.random.uniform(-3, 3)
        while True:
            x2 = np.random.uniform(-2, 2)
            print(0.5*(x2**2)-np.cos(x1))
            if 0.5*(x2**2)-np.cos(x1) < 0.99:
                break
        x0 = np.array([x1, x2])
        sol = solve_ivp(fun, t_span, x0, t_eval=np.linspace(*t_span, 100), vectorized=True)

        colors = cm.jet(sol.t / sol.t.max())
        plt.scatter(sol.y[0], sol.y[1], c=colors)
        plt.plot(sol.y[0], sol.y[1])
    plt.show()


def generate_pendulum(time_stop, n_points, **kwargs):

    while True:
        fun = get_fun()
        t_span = (0, time_stop)
        x1 = np.random.uniform(-3.1, 3.1)
        bound = np.sqrt(2*(np.cos(x1)+0.99))
        if not np.isnan(bound):
            break
    x2 = np.random.uniform(-bound, bound)

    x_init = tuple([x1, x2])
    sol = solve_ivp(fun, t_span, x_init, t_eval=np.linspace(*t_span, n_points), vectorized=True)
    return x_init, sol.y.T[None, :, :]