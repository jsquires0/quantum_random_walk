import sys, os, argparse
import numpy as np
import cirq as cq
import matplotlib.pyplot as plt

class YGate(cq.Gate):
    # This is an attempt at implementing a custom gate operator, to be used
    # as a coin. Doesn't work... unclear why.
    def __init__(self):
        super(YGate, self)

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [1.0,  1.0j],
            [1.0j, 1.0]
        ]) / np.sqrt(2)

class QuantumRandomWalk():
    def __init__(self, 
                 **kwargs):
        """
        Expects a dictionary containing
            n_qbits: int number of qubits for position registers
            max_steps: int number of steps in each walk
            n_walks: int number of walks used to create distribution
            coin: 'Hadamard' or 'Y' ('Y' doesn't work as intended)
            spin: 'up', 'down', or 'up+idown'
            classical: if True, a measurement is made after each step, rather
              than only after the final step of the walk
        """
        self.__dict__.update(kwargs)

        # position register can hold values from 0 to 2^N-1
        self.pos_register = [cq.LineQubit( q) for q in range(0, self.n_qbits)]
        self.coin_register = cq.LineQubit( self.n_qbits)
        self.walk = cq.Circuit()

        # define 'Y' gate
        self.Y_gate = YGate()

        return

    def initialize_registers(self, spin):
        """ Initializes the coin and position registers. Walk starts halfway 
        between 0 and 2^N-1. 

        Args:
            spin: string representing the chosen initial state
        """

        # initialize position in center
        self.walk.append(
            cq.X.on(self.pos_register[0])
        )
        # initialize spin
        if spin == 'up':
            # default spin is up (0)
            pass
        
        elif spin == 'down':
            # flip the spin 
            self.walk.append(
                cq.X.on(self.coin_register)
            )

        elif spin == 'up+idown':
            # (|up> + i|down>)/sqrt(2)

            # create superposition with H
            self.walk.append(
                cq.H.on(self.coin_register)
            )
            # make |down> component imaginary
            self.walk.append(
                (cq.Z**0.5).on(self.coin_register)
            )
            
        return

    def apply_coin(self, coin):
        """ Applies coin operator to coin register. 

        Args:
            coin: string representing the coin operator. Supports 'Hadamard' 
              or 'Y'.
        """
        if coin == 'Hadamard':
            self.walk.append(
                cq.H.on(self.coin_register)
            )
        elif coin == 'Y':
            self.walk.append(
                self.Y_gate.on(self.coin_register)
            )
        else:
            print('Bad coin')
            sys.exit()
        return 

    def apply_shift(self):
        """ Increments/decrements the position registers conditioned on the 
        value of the coin register. The logic of both circuits is always processed,
        but only one of increment/decrement occurs due to conditioning on the coin.
        """

        # This X gate is necessary due to two competing conventions:  
        # 1) we want to increment if spin is up
        # 2) spin up is |0>.  
        # If coin is initially |0> (up), applying NOT will allow 
        # n-control-qubit toffolis in the increment circuit to activate
        self.walk.append(
            cq.X.on(self.coin_register)
        )

        # Now, for each qubit in position register, apply n-control toffoli
        # targeting qubit, controlled by all less significant qubits + coin
        pos_regs = [i for i in range(self.n_qbits-1, -1, -1)] #ordered least to most significant
        for r in pos_regs:
            # define target
            t_qbit = cq.LineQubit( r)
            # collect all less significant and coin
            c_qbits = []
            for q in range(self.n_qbits, r, -1):
                c_qbits.append(cq.LineQubit(q))

            self.walk.append(
                cq.X.controlled(len(c_qbits)).on(*c_qbits, t_qbit)
            )
            # apply NOT so that, if bit was flipped by n-toffoli, we can
            # flip next bit if needed by carry operation
            self.walk.append(
                cq.X.on(t_qbit)
            )

        # Already applied one NOT to coin. Apply a second NOT to restore initial
        # state and ensure only increment OR decrement runs, not both
        self.walk.append(
            cq.X.on(self.coin_register)
        )

        # now decrement
        # If coin register wasn't holding |1> then only the NOT gates below
        # will affect output (undoing the NOTs applied in each step of the
        # increment circuit).
        pos_regs.reverse()
        for r in pos_regs: # ordered most to least significant
            # define target
            t_qbit = cq.LineQubit( r)
            # collect all more significant position regs and coin
            c_qbits = []
            for q in range(self.n_qbits, r, -1):
                c_qbits.append(cq.LineQubit( q))
            
            self.walk.append(
                cq.X.on(t_qbit)
            )
            self.walk.append(
                cq.X.controlled(len(c_qbits)).on(*c_qbits, t_qbit)
            )

        return 

    def measure(self, key = 'all'):
        """ Measures the circuit position registers to determine 
        particle location.

        Args:
          key: string labeled the measurement outcome
        """
        self.walk.append(
            cq.measure(*self.pos_register, key = key)
        )
        return

    def execute_classical_walk(self):
        """ Executs classical random walk, by measuring after each step.

        Returns:
            distribution of particle location
        """
        steps = 0
        while steps < self.max_steps:
            # one iteration is coin flip, shift, measure
            self.apply_coin(self.coin)
            self.apply_shift()
            self.measure(key=str(steps))
            steps += 1

        self.measure()
        sim = cq.Simulator()
        output = sim.run(self.walk, repetitions=self.n_walks)

        return output.histogram(key='all')

    def execute_quantum_walk(self):
        """ Executes quantum random walk.

        Returns:
            distribution of particle locations
        """
        steps = 0
        while steps < self.max_steps:
            self.apply_coin(self.coin)
            self.apply_shift()
            steps += 1

        self.measure()
        sim = cq.Simulator()
        output = sim.run(self.walk, repetitions=self.n_walks)

        return output.histogram(key='all')

    def random_walk(self, classical):
        """ Executes n_walk random walks, creating a distribution of particle
        locations. 

        Args:
            classical: if True, a measurement is made after every time step
        Returns:
            distribution of particle location
        """

        self.initialize_registers(self.spin)
        
        if classical:
            return self.execute_classical_walk()
        else:
            return self.execute_quantum_walk()

    def plot_distribution(self, distribution):
        """ Plots histogram data for classical/quantum random walk. Saves
        to file in run directory.

        Args:
            distribution: Counter object of particle location frequencies
        """
        # label as quantum/classical
        symbol = 'Q'
        if self.classical:
            symbol = 'C'
        title = f'{self.max_steps}-step {symbol}RW over N = {self.n_walks} sims for init. spin = {self.spin}'
        
        # generate histogram
        cq.vis.plot_state_histogram(distribution, 
                                    plt.subplot(), 
                                    title=title,
                                    ylabel = 'Counts',
                                    xlabel = 'x',
                                    )
        # clean up and label plot
        lims = max(distribution.keys()) - min(distribution.keys())
        n_ticks = max(2, round(lims/10))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(n_ticks))

        # display briefly and save to file
        filename = f'{symbol}RW' + f'_{self.spin}_' + 'hist.png'
        plt.savefig(filename)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        return

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Quantum Random Walk Simulator')
    parser.add_argument('-N','--n_qbits', 
            help='Number of qubits', 
            required=False, 
            type = int,
            default = 6)
    parser.add_argument('-t','--max_steps', 
            help='Number of steps', 
            required=False, 
            type = int,
            default = 25)
    parser.add_argument('-w','--n_walks', 
            help='Number of simulations', 
            required=False, 
            type = int,
            default = 1000)
    parser.add_argument('-co','--coin', 
            help='Coin operator: "Hadamard" or "Y"', 
            required=False, 
            type = str,
            default = "Hadamard")
    parser.add_argument('-s','--spin', 
            help='Initial spin options: "up", "down", or "up+idown"', 
            required=False, 
            type = str,
            default = "up")
    parser.add_argument('--classical', dest='classical', action='store_true')
    parser.add_argument('--quantum', dest='classical', action='store_false')
    parser.set_defaults(classical=False)
    args = vars(parser.parse_args())

    # initialize, execute, and plot the walk simulations
    QRW = QuantumRandomWalk(**args)
    dist = QRW.random_walk(classical = QRW.classical)
    QRW.plot_distribution(dist)
