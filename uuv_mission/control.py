from dataclasses import dataclass

@dataclass
class PDController:
    """Discrete-time proportional-derivative (PD) controller.

    Implements: u[t] = KP * e[t] + KD * (e[t] - e[t-1]) / dt
    where e[t] = r[t] - y[t].

    Notes:
    - If your simulation uses a unit time step (dt = 1), the equation reduces to
      u[t] = KP * e[t] + KD * (e[t] - e[t-1]).
    - The controller stores the previous error (err_prev) as internal state and
      provides a reset() method to clear it between runs.
    - The default gains are KP=0.15 and KD=0.6.
    """

    KP: float = 0.15
    KD: float = 0.6
    err_prev: float = 0.0

    def reset(self) -> None:
        """Reset the internal controller state (previous error)."""
        self.err_prev = 0.0

    def __call__(self, ref_t: float, y_t: float, dt: float = 1.0) -> float:
        """Compute the control action for the current time step.

        Args:
            ref_t: Reference value at time t.
            y_t: Measured output (depth) at time t.
            dt: Discrete time step (must be > 0, defaults to 1.0). If your plant
                uses a different time step, pass that here so the derivative term is
                scaled correctly.

        Returns:
            Control action u_t.
        """

        err_t = ref_t - y_t

        if dt <= 0:
            raise ValueError(f"dt must be positive; received dt={dt}")

        deriv = (err_t - self.err_prev) / dt

        u_t = self.KP * err_t + self.KD * deriv

        # Update internal state
        self.err_prev = err_t

        return u_t
