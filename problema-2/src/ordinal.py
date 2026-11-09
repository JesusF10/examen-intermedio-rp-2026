"""Logit ordinal propio: NLL estable, gradiente analítico y umbrales ordenados."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import check_grad, minimize
from scipy.special import expit, log_expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

Array = NDArray[np.float64]


def umbrales(u: Array) -> Array:
    return np.r_[u[0], u[0] + np.cumsum(np.exp(u[1:]))]


def inicializar(y: NDArray[np.int64], k: int, d: int) -> Array:
    conteos = np.bincount(y, minlength=k) + 0.5
    theta = logit(np.cumsum(conteos)[:-1] / conteos.sum())
    return np.r_[np.zeros(d), theta[0], np.log(np.diff(theta))]


def nll_grad(
    parametros: Array, x: Array, y: NDArray[np.int64], k: int, alpha: float = 0.0
) -> tuple[float, Array]:
    """NLL media + alpha*||w||²/2 y gradiente respecto a (w,u).

    theta_1=u_1 y theta_j=u_1+sum(exp(u_l), l=2..j). No hay intercepto en x.
    """
    d, n = x.shape[1], len(y)
    w, u = parametros[:d], parametros[d:]
    theta = umbrales(u)
    extendidos = np.r_[-np.inf, theta, np.inf]
    eta = x @ w
    a, b = extendidos[y + 1] - eta, extendidos[y] - eta
    primero, ultimo = y == 0, y == k - 1
    medio = ~(primero | ultimo)
    logp = np.empty(n)
    logp[primero] = log_expit(a[primero])
    logp[ultimo] = log_expit(-b[ultimo])
    # log(sigmoid(a)-sigmoid(b)), sin restar probabilidades casi iguales.
    logp[medio] = (
        log_expit(a[medio])
        + log_expit(-b[medio])
        + np.log(-np.expm1(-(theta[y[medio]] - theta[y[medio] - 1])))
    )
    residual = 1 - expit(a) - expit(b)
    grad_w = x.T @ residual / n + alpha * w
    razon_a = np.exp(log_expit(a) + log_expit(-a) - logp)
    razon_b = np.exp(log_expit(b) + log_expit(-b) - logp)
    grad_theta = (
        np.bincount(y[~ultimo], weights=-razon_a[~ultimo], minlength=k - 1)
        + np.bincount(y[~primero] - 1, weights=razon_b[~primero], minlength=k - 1)
    ) / n
    acumulado = np.cumsum(grad_theta[::-1])[::-1]
    grad_u = np.r_[acumulado[0], np.exp(u[1:]) * acumulado[1:]]
    return float(-np.mean(logp) + alpha * np.dot(w, w) / 2), np.r_[grad_w, grad_u]


class ModeloOrdinal(ClassifierMixin, BaseEstimator):
    def __init__(
        self, alpha: float = 0.0, regla: str = "mediana", max_iter: int = 1500
    ):
        self.alpha = alpha
        self.regla = regla
        self.max_iter = max_iter

    def fit(self, x: Array, y: NDArray[np.int64]) -> ModeloOrdinal:
        self.classes_, indices = np.unique(y, return_inverse=True)
        k, d = len(self.classes_), x.shape[1]
        if k < 2 or self.alpha < 0 or not np.all(np.isfinite(x)):
            raise ValueError("Se requieren >=2 clases, alpha no negativo y X finita.")
        inicial = inicializar(indices, k, d)
        resultado = minimize(
            nll_grad,
            inicial,
            args=(x, indices, k, self.alpha),
            jac=True,
            method="L-BFGS-B",
            bounds=[(None, None)] * (d + 1) + [(-12, 12)] * (k - 2),
            options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-7},
        )
        self.parametros_ = resultado.x
        self.coef_ = resultado.x[:d]
        self.umbrales_ = umbrales(resultado.x[d:])
        self.n_iter_ = int(resultado.nit)
        self.convergencia_ = bool(resultado.success)
        self.gradiente_max_ = float(np.max(np.abs(resultado.jac)))
        self.mensaje_ = str(resultado.message)
        if not self.convergencia_ or self.gradiente_max_ > 1e-4:
            raise RuntimeError(
                f"Ordinal sin convergencia suficiente: {self.mensaje_}, grad={self.gradiente_max_}"
            )
        if np.any(np.abs(resultado.x[d + 1 :]) > 11.9):
            raise RuntimeError(
                "Umbral en la frontera numérica: revisar parametrización."
            )
        return self

    def predict_proba(self, x: Array) -> Array:
        acumuladas = expit(self.umbrales_[None, :] - (x @ self.coef_)[:, None])
        return np.diff(np.c_[np.zeros(len(x)), acumuladas, np.ones(len(x))], axis=1)

    def predict(self, x: Array) -> NDArray[np.int64]:
        p = self.predict_proba(x)
        if self.regla == "moda":
            indices = np.argmax(p, axis=1)
        elif self.regla == "mediana":
            indices = np.sum(np.cumsum(p, axis=1) < 0.5, axis=1)
        else:
            raise ValueError("Regla ordinal inválida.")
        return self.classes_[indices]


def verificar(x: Array, y: NDArray[np.int64], carpeta: Path) -> ModeloOrdinal:
    """Usa solamente entrenamiento y contrasta el ajuste sin penalizar."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    inicio = perf_counter()
    escalador = StandardScaler().fit(x)
    z = escalador.transform(x)
    clases, indices = np.unique(y, return_inverse=True)
    k, d = len(clases), z.shape[1]
    rng = np.random.default_rng(np.random.SeedSequence([221201513, 2, 1]))
    errores = []
    for alpha in (0.0, 0.1):
        p = inicializar(indices, k, d) + rng.normal(0, 0.05, d + k - 1)
        error = check_grad(
            lambda t: nll_grad(t, z[:128], indices[:128], k, alpha)[0],
            lambda t: nll_grad(t, z[:128], indices[:128], k, alpha)[1],
            p,
            epsilon=1e-6,
        )
        if error > 1e-4:
            raise ArithmeticError(f"check_grad falló: {error}")
        errores.append({"alpha": alpha, "norma_diferencia": float(error)})
    with threadpool_limits(limits=1):
        propio = ModeloOrdinal().fit(z, y)
        referencia = OrderedModel(indices, z, distr="logit").fit(
            method="bfgs", maxiter=2000, gtol=1e-7, disp=False
        )
    w_sm = np.asarray(referencia.params[:d])
    theta_sm = referencia.model.transform_threshold_params(referencia.params)[1:-1]
    diferencia_w = float(np.max(np.abs(propio.coef_ - w_sm)))
    diferencia_theta = float(np.max(np.abs(propio.umbrales_ - theta_sm)))
    diferencia_prob = float(
        np.max(
            np.abs(
                propio.predict_proba(z) - referencia.model.predict(referencia.params)
            )
        )
    )
    if (
        not referencia.mle_retvals["converged"]
        or diferencia_w > 5e-4
        or diferencia_theta > 1e-3
        or diferencia_prob > 1e-4
    ):
        raise ArithmeticError("No coincide con OrderedModel dentro de las tolerancias.")
    carpeta.mkdir(parents=True, exist_ok=True)
    resultado = {
        "check_grad": errores,
        "epsilon": 1e-6,
        "tolerancia_check_grad": 1e-4,
        "max_diferencia_w": diferencia_w,
        "tolerancia_w": 5e-4,
        "max_diferencia_theta": diferencia_theta,
        "tolerancia_theta": 1e-3,
        "max_diferencia_probabilidad": diferencia_prob,
        "tolerancia_probabilidad": 1e-4,
        "convergencia_propia": propio.convergencia_,
        "convergencia_statsmodels": bool(referencia.mle_retvals["converged"]),
        "gradiente_propio_max": propio.gradiente_max_,
        "iteraciones_propio": propio.n_iter_,
        "nll_media_propia": nll_grad(propio.parametros_, z, indices, k)[0],
        "nll_media_statsmodels": float(-referencia.llf / len(y)),
        "coeficientes_propios": propio.coef_.tolist(),
        "coeficientes_statsmodels": w_sm.tolist(),
        "umbrales_propios": propio.umbrales_.tolist(),
        "umbrales_statsmodels": theta_sm.tolist(),
        "segundos": perf_counter() - inicio,
    }
    (carpeta / "verificacion-ordinal.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(resultado, indent=2, ensure_ascii=False), flush=True)
    return propio


def main() -> None:
    from .datos import RESULTADOS, cargar_entrenamiento

    x, y, _ = cargar_entrenamiento()
    verificar(x, y, RESULTADOS / "tablas")


if __name__ == "__main__":
    main()
