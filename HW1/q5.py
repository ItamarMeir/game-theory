import random
from dataclasses import dataclass
from typing import List, Tuple

# matplotlib is imported lazily inside the plotting function to avoid requiring
# it at module import time (fixes static import resolution in some editors).


@dataclass(frozen=True)
class GMPG:
    HH: int
    HT: int
    TH: int
    TT: int

    def __post_init__(self):
        if not all(v > 0 for v in (self.HH, self.HT, self.TH, self.TT)):
            raise ValueError("All parameters must be positive integers.")

    @staticmethod
    def random(low: int = 11, high: int = 20) -> "GMPG":
        return GMPG(*(random.randint(low, high) for _ in range(4)))

    def payoff(self, row: str, col: str) -> Tuple[int, int]:
        """
        row: 'H' or 'T' (player 1)
        col: 'H' or 'T' (player 2)
        Returns (u_row, u_col)
        """
        if row == 'H' and col == 'H':
            return (self.HH, -self.HH)
        if row == 'H' and col == 'T':
            return (-self.HT, self.HT)
        if row == 'T' and col == 'H':
            return (-self.TH, self.TH)
        if row == 'T' and col == 'T':
            return (self.TT, -self.TT)
        raise ValueError("Strategies must be 'H' or 'T'.")

    def __repr__(self):
        return f"GMPG(HH={self.HH}, HT={self.HT}, TH={self.TH}, TT={self.TT})"


@dataclass
class Engine:
    game: GMPG
    T: int  # number of rounds

    def __post_init__(self):
        if self.T <= 0:
            raise ValueError("T must be a positive integer.")

    # ---------- Best responses ----------

    def best_response_row(self, current_row: str, col: str) -> str:
        """
        Player 1 best response to column's action.
        Tie-breaking: keep current action if it's a best response.
        """
        u_H, _ = self.game.payoff('H', col)
        u_T, _ = self.game.payoff('T', col)
        best_val = max(u_H, u_T)

        # If current action is already a best response, don't change
        if current_row == 'H' and u_H == best_val:
            return 'H'
        if current_row == 'T' and u_T == best_val:
            return 'T'

        # Otherwise switch to the better one
        return 'H' if u_H > u_T else 'T'

    def best_response_col(self, row: str, current_col: str) -> str:
        """
        Player 2 best response to row's action.
        Tie-breaking: keep current action if it's a best response.
        """
        _, u_H = self.game.payoff(row, 'H')
        _, u_T = self.game.payoff(row, 'T')
        best_val = max(u_H, u_T)

        if current_col == 'H' and u_H == best_val:
            return 'H'
        if current_col == 'T' and u_T == best_val:
            return 'T'

        return 'H' if u_H > u_T else 'T'

    # ---------- BR dynamics simulation ----------

    def simulate(self):
        """
        Simulate best response dynamics for T rounds.
        Round 0: (H, H) is fixed and counted.
        Odd t:   column moves.
        Even t>0: row moves.
        Returns:
            rows, cols: list of actions per round
            u1, u2: utilities per round
        """
        # Start from (H, H)
        row_action = 'H'
        col_action = 'H'

        rows: List[str] = [row_action]
        cols: List[str] = [col_action]
        u1: List[int] = []
        u2: List[int] = []

        # payoff at t=0
        p1, p2 = self.game.payoff(row_action, col_action)
        u1.append(p1)
        u2.append(p2)

        for t in range(1, self.T):
            if t % 2 == 1:
                # odd t: column player 2 moves
                col_action = self.best_response_col(row_action, col_action)
            else:
                # even t>0: row player 1 moves
                row_action = self.best_response_row(row_action, col_action)

            rows.append(row_action)
            cols.append(col_action)

            p1, p2 = self.game.payoff(row_action, col_action)
            u1.append(p1)
            u2.append(p2)

        return rows, cols, u1, u2

    @staticmethod
    def prefix_average(values: List[float]) -> List[float]:
        """
        Compute running averages: avg[0] = v0,
        avg[t] = (v0 + ... + vt)/(t+1)
        """
        avg = []
        total = 0.0
        for i, v in enumerate(values):
            total += v
            avg.append(total / (i + 1))
        return avg
    def simulate_and_plot(self, title_suffix: str = ""):
        """
        Run BR dynamics and plot average utilities up to each t, for both players.
        """
        # we don't need the full path history here, only utilities; use placeholders
        _, _, u1, u2 = self.simulate()
        avg_u1 = self.prefix_average(u1)
        avg_u2 = self.prefix_average(u2)

        rounds = list(range(self.T))

        # import matplotlib lazily to avoid module-level import issues in some editors
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))

        # plot both players on the same axes with different colors and legend
        ax.plot(rounds, avg_u1, label="P1 avg utility", color="C0")
        ax.plot(rounds, avg_u2, label="P2 avg utility", color="C1")
        ax.set_xlabel("Round t")
        ax.set_ylabel("Avg utility")
        ax.grid(True)
        ax.legend()

        fig.suptitle(f"Best-response dynamics in {self.game} {title_suffix}")
        plt.tight_layout()
        plt.show()
        plt.show()


if __name__ == "__main__":
    T = 1000

    for i in range(3):
        game = GMPG.random()
        print(f"Instance {i+1}: {game}")

        eng = Engine(game=game, T=T)
        eng.simulate_and_plot(title_suffix=f"(instance {i+1})")
