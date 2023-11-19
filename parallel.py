import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.opflow import CircuitSampler, PauliOp, MatrixEvolution, Suzuki
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus, CX, CZ, Swap
from qiskit.circuit import Parameter
from qiskit import (
    QuantumCircuit,
    Aer,
    assemble,
    QuantumRegister,
    IBMQ,
    execute,
    transpile,
)
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import HamiltonianGate
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import random
from qiskit.tools.visualization import circuit_drawer
import warnings
import math

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (20, 10)


def get_prob(counts, N, numQubits):
    keys = list(counts.keys())
    keys.sort()
    counts = {i: counts[i] for i in keys}
    for key in counts:
        counts[key] /= N
    counts = list(counts.items())
    prob = [0 for i in range(2**numQubits)]
    # prob = np.zeros(2**numQubits) #[0 for i in range(2**numQubits)]
    for i in range(len(counts)):
        prob[int(counts[i][0], 2)] = counts[i][1]
    return prob


def L_diag(numQubits):
    if numQubits == 1:
        return -2 * I
    return L_diag(numQubits - 1) ^ I


def p(gate, size):
    data = Operator(gate).data
    for i in range(size):
        print(data[i].real)
    print("")


top_left = (I + Z) / 2
top_right = (X + (0 + 1j) * Y) / 2
bot_left = (X - (0 + 1j) * Y) / 2
bot_right = (I - Z) / 2


def EmptyGate(numQubits):
    gate = I - I
    for i in range(numQubits - 1):
        gate = gate ^ I
    return gate


def oneAt(i, j, numQubits):
    gate = EmptyGate(1)
    if i < 2**numQubits / 2 and j < 2**numQubits / 2:
        gate = top_left
    if i < 2**numQubits / 2 and j >= 2**numQubits / 2:
        gate = top_right
    if i >= 2**numQubits / 2 and j < 2**numQubits / 2:
        gate = bot_left
    if i >= 2**numQubits / 2 and j >= 2**numQubits / 2:
        gate = bot_right
    if numQubits == 1:
        return gate
    return gate ^ oneAt(
        i % (2**numQubits / 2), j % (2**numQubits / 2), numQubits - 1
    )


def L_cyclic(numQubits):
    gate = EmptyGate(numQubits)
    for i in range(2**numQubits):
        for j in range(2**numQubits):
            if (i, j) == (0, 2**numQubits - 1) or (j, i) == (0, 2**numQubits - 1):
                gate += oneAt(i, j, numQubits)
            if abs(i - j) == 1:
                gate += oneAt(i, j, numQubits)
    return L_diag(numQubits) + gate


def L_complete(numQubits):
    gate = EmptyGate(numQubits)
    for i in range(2**numQubits):
        for j in range(2**numQubits):
            if i == j:
                gate -= (2**numQubits - 1) * oneAt(i, j, numQubits)
            else:
                gate += oneAt(i, j, numQubits)
    return gate


from qiskit.providers.aer import StatevectorSimulator
from qiskit.providers.aer import AerSimulator
from scipy.linalg import expm


def complete(numQubits, psi, g, delta_t, delta_x, V_diag, n, N):
    gamma = 1 / (2 * delta_x**2)
    L = L_complete(numQubits)
    U = (-1 * gamma * L * delta_t).exp_i()  # u.exp_i() returns e^(-iu)
    U = Operator(U)

    data = np.zeros((n + 1, len(psi)))
    prob = np.square(psi)
    data[0] = prob  # [[psi[i]**2 for i in range(len(psi))]]

    # step 1
    qc = QuantumCircuit(numQubits)
    qc.initialize(psi, qc.qubits)
    qc.append(U, qc.qubits)

    # step 4

    for j in range(n):  # step 5&
        # step 6
        V = [[0 for k in range(2**numQubits)] for i in range(2**numQubits)]
        for i in range(2**numQubits):
            V[i][i] = np.exp((0 + 1j) * (g * prob[i] - V_diag[i]) * delta_t)
        # V.append(Operator(Op)) #append V_j
        # qc.append(U,qc.qubits)
        qc.append(Operator(V), qc.qubits)

        # measure ψ(j + 1)
        copy = qc.copy()
        copy.measure_all()
        backend_sim = Aer.get_backend("qasm_simulator")
        job_sim = backend_sim.run(transpile(copy, backend_sim), shots=N)
        result_sim = job_sim.result()
        prob = get_prob(result_sim.get_counts(copy), N, numQubits)
        data[j + 1] = prob  # data.append(prob) #
        # print(prob)

        # step 7
        qc.append(U, qc.qubits)

    return data.T


def complete_get_statevector(numQubits, psi, g, delta_t, delta_x, V_diag, n, N):
    gamma = 1 / (2 * delta_x**2)
    L = L_complete(numQubits)
    U = (-1 * gamma * L * delta_t).exp_i()
    U = Operator(U)

    data = np.zeros((n + 1, len(psi)))
    prob = np.square(psi)
    data[0] = prob

    statevector_list = [psi]

    for j in range(n):  # step 5&
        if j % 100 == 0:
            print(j)
        qc = QuantumCircuit(numQubits)
        qc.initialize(psi, qc.qubits)
        qc.append(U, qc.qubits)

        V = [[0 for k in range(2**numQubits)] for i in range(2**numQubits)]
        for i in range(2**numQubits):
            V[i][i] = np.exp((0 + 1j) * (g * prob[i] - V_diag[i]) * delta_t)
        qc.append(Operator(V), qc.qubits)

        copy = qc.copy()
        copy.measure_all()
        backend_sim = Aer.get_backend("qasm_simulator")
        job_sim = backend_sim.run(transpile(copy, backend_sim), shots=N)
        result_sim = job_sim.result()
        prob = get_prob(result_sim.get_counts(copy), N, numQubits)
        data[j + 1] = prob

        statevector_circuit = copy.copy()  # This creates a new copy of the circuit
        statevector_circuit.remove_final_measurements()  # This removes the measurements from the copied circuit

        # 2. Use the statevector_simulator to get the statevector
        backend_statevector = Aer.get_backend("statevector_simulator")
        job_statevector = backend_statevector.run(
            transpile(statevector_circuit, backend_statevector)
        ).result()
        statevector = job_statevector.get_statevector()
        statevector_list.append(statevector)
        psi = statevector

    return np.array(statevector_list)


def complete_get_statevector_no_trotter(
    numQubits, psi, g, delta_t, delta_x, V_diag, n, N
):
    gamma = 1 / (2 * delta_x**2)
    # L = L_complete(numQubits)
    L = np.ones((8, 8), dtype=complex)
    np.fill_diagonal(L, -7 + 0j)

    data = np.zeros((n + 1, len(psi)))
    prob = np.square(psi)
    data[0] = prob

    statevector_list = [psi]

    for j in range(n):
        if j % 100 == 0:
            print(j)
        qc = QuantumCircuit(numQubits)
        qc.initialize(psi, qc.qubits)  # type: ignore

        H = -gamma * L + np.diag(V_diag) - np.diag(g * prob)

        gate = Operator(expm(-1j * H * delta_t)).to_instruction()
        qc.append(gate, qc.qubits)

        copy = qc.copy()
        copy.measure_all()
        backend_sim = Aer.get_backend("qasm_simulator")
        job_sim = backend_sim.run(transpile(copy, backend_sim), shots=N)
        result_sim = job_sim.result()
        prob = get_prob(result_sim.get_counts(copy), N, numQubits)
        data[j + 1] = prob

        statevector_circuit = copy.copy()  # This creates a new copy of the circuit
        statevector_circuit.remove_final_measurements()  # This removes the measurements from the copied circuit

        # 2. Use the statevector_simulator to get the statevector
        backend_statevector = Aer.get_backend("statevector_simulator")
        job_statevector = backend_statevector.run(
            transpile(statevector_circuit, backend_statevector)
        ).result()
        statevector = job_statevector.get_statevector()
        statevector_list.append(statevector)
        psi = statevector

    return np.array(statevector_list).T


from scipy.integrate import solve_ivp


def deriv(t, psi, gamma, numQubits, V, g):
    L = np.zeros(V.shape)
    for i in range(len(L)):
        for j in range(len(L)):
            L[i][j] = -(2**numQubits - 1) if i == j else 1
    K = np.diag(np.abs(psi) ** 2)
    psidot = np.matmul((-gamma * L + V - g * K) * (-1j), psi)
    psidot.tolist()
    return psidot


def exact_complete(numQubits, psi_init, gamma, V_diag, g, n, max_step):
    V = np.diag(V_diag)

    p = (gamma, numQubits, V, g)
    t0, tf = 0, n

    psi = tuple([complex(psi_init[i]) for i in range(len(psi_init))])
    soln = solve_ivp(deriv, (t0, tf), psi, args=p, max_step=max_step)

    for j in range(len(soln.y)):
        for i in range(len(soln.y[j])):
            soln.y[j][i] = np.absolute(soln.y[j][i]) ** 2  # |ψ|^2

    return soln.t, soln.y


def exact_complete_get_psi(numQubits, psi_init, gamma, V_diag, g, n, max_step):
    V = np.diag(V_diag)

    p = (gamma, numQubits, V, g)
    t0, tf = 0, n

    psi = tuple([complex(psi_init[i]) for i in range(len(psi_init))])
    soln = solve_ivp(deriv, (t0, tf), psi, args=p, max_step=max_step)

    return soln.t, soln.y

def compare(psi, g, delta_t, delta_x, T, N):
    psi = [1 / math.sqrt(8) for i in range(8)]
    V_diag = [1, 0, 0, 0, 0, 0, 0, 0]
    n = int(T / delta_t)
    data = np.zeros((N, n + 1))
    for i in range(N):
        print(str(i) + "/" + str(N))
        data[i] = (
            np.abs(
                complete_get_statevector_no_trotter(
                    3, psi, g, delta_t, delta_x, V_diag, n, 1024 * 4
                )[0]
            )
            ** 2
        )
    mu_hat, stdev_hat = np.mean(data, axis=0), np.std(data, axis=0)
    return mu_hat, stdev_hat

exact_t = np.load("gamma=2/exact_t.npy")
exact = np.load("gamma=2/exact.npy")

def f(psi, g, delta_t, delta_x, T, N):
    mu_hat, stdev_hat = compare(psi, g, delta_t, delta_x, T, N)
    err = np.zeros_like(mu_hat)
    exact_t = np.load("gamma=2/exact_t.npy")
    exact = np.load("gamma=2/exact.npy")
    exact_plot = []
    exact_plot_t = []
    for i in range(2, len(err)):
        t = round(i * delta_t, 3)
        exact_plot_t.append(t)
        err[i] = np.abs(exact[0][int(t / 0.001)] - mu_hat[i]) / (stdev_hat[i] / np.sqrt(N))
        exact_plot.append(exact[0][int(t / 0.001)])
    return err

start_time = time.time()
from concurrent.futures import ProcessPoolExecutor
import os

# Function to be executed in parallel
def calculate_and_save(dx):
    t = 0.05
    err = f(psi=[1 / math.sqrt(8) for i in range(8)], g=1, delta_t=t, delta_x=dx, T=50, N=500)
    if not os.path.exists("delta_xs"):
        os.makedirs("delta_xs")
    filename = f"delta_xs/f(T=500,delta_x={round(dx, 3)}, delta_t={round(t, 3)}).npy"
    np.save(filename, err)
    return err


# List of delta_x values
delta_xs = np.arange(0.1, 1.1, 0.1)

# Execute the tasks in parallel using ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    executor.map(calculate_and_save, delta_xs)

print("--- %s para ---" % (time.time() - start_time))

# # The rest of your code that needs tstart_time = time.time()o run after parallel execution
