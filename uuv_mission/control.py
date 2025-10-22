from dataclasses import dataclass

@dataclass
class PDController:
    """Discrete-time proportional-derivative (PD) controller.

    Implements: u[t] = KP * e[t] + KD * (e[t] - e[t-1]) / dt
    where e[t] = r[t] - y[t].

    Notes:
    - If your simulation uses a unit time step (dt = 1), the equation reduces to
      u[t] = KP * e[t] + KD * (e[t] - e[t-1]).
    - The controller stores the previous error (e_prev) as internal state and
      provides a reset() method to clear it between runs.
    - The default gains are KP=0.15 and KD=0.6.
    """

    KP: float = 0.15
    KD: float = 0.6
    e_prev: float = 0.0

    def reset(self) -> None:
        """Reset the internal controller state (previous error)."""
        self.e_prev = 0.0

    def __call__(self, r_t: float, y_t: float, dt: float = 1.0) -> float:
        """Compute the control action for the current time step.

        Args:
            r_t: Reference value at time t.
            y_t: Measured output (depth) at time t.
            dt: Discrete time step (must be > 0, defaults to 1.0). If your plant
                uses a different time step, pass that here so the derivative term is
                scaled correctly.

        Returns:
            Control action u_t.
        """
        # Current error
        e_t = r_t - y_t

        if dt <= 0:
            raise ValueError(f"dt must be positive; received dt={dt}")

        # Derivative term
        deriv = (e_t - self.e_prev) / dt

        # PD control law
        u_t = self.KP * e_t + self.KD * deriv

        # Update internal state
        self.e_prev = e_t

        return u_t
