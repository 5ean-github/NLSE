import time
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus, CX, CZ, Swap
from qiskit import (
    QuantumCircuit,
    Aer,
    execute,
    transpile,
)
from qiskit.quantum_info.operators import Operator, Pauli
import random
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
    return gate ^ oneAt(i % (2**numQubits / 2), j % (2**numQubits / 2), numQubits - 1)


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

# not used
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

    prob = np.square(psi)

    statevector_list = [psi]

    for j in range(n):
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

        # statevector_circuit = copy.copy()  # This creates a new copy of the circuit
        # statevector_circuit.remove_final_measurements()  # This removes the measurements from the copied circuit
        copy.remove_final_measurements()
        # 2. Use the statevector_simulator to get the statevector
        backend_statevector = Aer.get_backend("statevector_simulator")
        job_statevector = backend_statevector.run(
            transpile(copy, backend_statevector)
        ).result()
        statevector = job_statevector.get_statevector()
        statevector_list.append(statevector)
        psi = statevector

    return np.array(statevector_list).T


def complete_get_statevector_no_trotter_no_meas(
    numQubits, psi, g, delta_t, delta_x, V_diag, n, N
):
    gamma = 1 / (2 * delta_x**2)
    # L = L_complete(numQubits)
    L = np.ones((8, 8), dtype=complex)
    np.fill_diagonal(L, -7 + 0j)

    prob = np.square(psi)

    statevector_list = [psi]

    for j in range(n):
        if j % 100 == 0:
            print(j)
        qc = QuantumCircuit(numQubits)
        qc.initialize(psi, qc.qubits)  # type: ignore

        H = -gamma * L + np.diag(V_diag) - np.diag(g * prob)

        gate = Operator(expm(-1j * H * delta_t)).to_instruction()
        qc.append(gate, qc.qubits)

        # 2. Use the statevector_simulator to get the statevector
        backend_statevector = Aer.get_backend("statevector_simulator")
        job_statevector = backend_statevector.run(
            transpile(qc, backend_statevector)
        ).result()
        statevector = job_statevector.get_statevector()
        statevector_list.append(statevector)
        prob = np.abs(statevector) ** 2
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


def compare(T, delta_t, N):
    psi = [1 / math.sqrt(8) for i in range(8)]
    V_diag = [1, 0, 0, 0, 0, 0, 0, 0]
    n = int(T / delta_t)
    data = np.zeros((N, n + 1))
    for i in range(N):
        print(str(i) + "/" + str(N))
        data[i] = (
            np.abs(
                complete_get_statevector_no_trotter(
                    3, psi, 1, delta_t, 1 / math.sqrt(2), V_diag, n, 256
                )[0]
            )
            ** 2
        )
    mu_hat, stdev_hat = np.mean(data, axis=0), np.std(data, axis=0)
    return mu_hat, stdev_hat


exact_t = np.load("exact_t.npy")
exact = np.load("exact.npy")


def f(T, delta_t, N):
    mu_hat, stdev_hat = compare(T, delta_t, N)
    err = np.zeros_like(mu_hat)
    exact_plot = []
    exact_plot_t = []
    for i in range(2, len(err)):
        t = round(i * delta_t, 3)
        exact_plot_t.append(t)
        err[i] = np.abs(exact[0][int(t / 0.001)] - mu_hat[i]) / (
            stdev_hat[i] / np.sqrt(N)
        )
        exact_plot.append(exact[0][int(t / 0.001)])
    return err


def L1(T, delta_t, N):
    sqrt_inv_8 = 1 / math.sqrt(8)
    sqrt_inv_2 = 1 / math.sqrt(2)
    psi = [sqrt_inv_8 for _ in range(8)]
    V_diag = [1, 0, 0, 0, 0, 0, 0, 0]
    n = int(T / delta_t)
    data = np.zeros((N, n + 1))
    err = np.zeros(n + 1)
    diff = np.zeros((N, n + 1))
    for i in range(N):
        print(str(i) + "/" + str(N))
        data[i] = (
            np.abs(
                complete_get_statevector_no_trotter(
                    3, psi, 1, delta_t, sqrt_inv_2, V_diag, n, 256
                )[0]
            )
            ** 2
        )
        for j in range(n + 1):
            t = round(j * delta_t, 3)
            diff[i][j] = np.abs(exact[0][int(t / 0.001)] - data[i][j])
    for j in range(n + 1):
        err[j] = np.mean(diff[:, j])
    return err


def optimized_L1(T, delta_t, N, M):
    sqrt_inv_8 = 1 / math.sqrt(8)
    sqrt_inv_2 = 1 / math.sqrt(2)
    psi = [sqrt_inv_8 for _ in range(8)]
    V_diag = [1, 0, 0, 0, 0, 0, 0, 0]
    n = int(T / delta_t)
    t_values = np.round(np.arange(0, T + 0.0001, delta_t), 3)
    exact_values = np.array([exact[0][int(t / 0.001)] for t in t_values])
    diffs = []
    for i in range(N):
        print(str(i) + "/" + str(N))
        data = (
            np.abs(
                complete_get_statevector_no_trotter(
                    3, psi, 1, delta_t, sqrt_inv_2, V_diag, n, M
                )[0]
            )
            ** 2
        )
        print(data.shape)
        diff = np.abs(exact_values - data)
        diffs.append(diff)

    diff_array = np.array(diffs)
    err = np.mean(diff_array, axis=0)
    return err



start_time = time.time()
from concurrent.futures import ProcessPoolExecutor


# Function to be executed in parallel
def calculate_and_save(t, measurements=4):
    T, delta_t, N = 50, t, 500
    M = measurements
    err = optimized_L1(T, delta_t, N, M)
    # plt.plot(err)
    # plt.xlabel("time")
    # plt.ylabel("f")
    # plt.savefig("N=500/delta_t=" + str(round(t, 3)) + ".png")
    filename = f"M={str(M)}/f(T=50,delta_t={str(round(delta_t,4))}).npy"
    os.makedirs(os.path.dirname("M=" + str(M) + "/"), exist_ok=True)
    np.save(filename, err)
    return err


# List of delta_t values
t_list = np.flip(np.arange(0.01, 0.1, 0.01))
# t_list = [0.04]
print(t_list)

M_list=[5,7,9,10,11,12,13,14,15]

with ProcessPoolExecutor() as executor:
    futures = []
    # for t in t_list:
    #     for M in M_list:
    #         futures.append(executor.submit(calculate_and_save, t, M))
    #futures.append(executor.submit(calculate_and_save, 0.0125, 5))
    futures.append(executor.submit(calculate_and_save, 0.025, 10))
    futures.append(executor.submit(calculate_and_save, 0.008, 4))
    futures.append(executor.submit(calculate_and_save, 0.016, 8))

print("--- %s para ---" % (time.time() - start_time))

# The rest of your code that needs tstart_time = time.time()o run after parallel execution
