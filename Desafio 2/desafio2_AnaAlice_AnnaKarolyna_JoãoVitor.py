# =============================================================================
# >>> DESAFIO 2 <<<
# =============================================================================

# =============================================================================
# >>> Integrantes do grupo: <<<
# João Vitor Feijó Asevedo - 12311BCC061
# Anna Karolyna Pereira Santos - 12221BCC046
# Ana Alice Cordeiro de Souza - 12211BCC028
# =============================================================================


import math
import torch
import torch.nn.functional as F


_N_CAMADAS_ATUAL = 16

ALPHA_ORIGINAL = 0.05
GANHO_BASE = math.sqrt(2.0 / (1.0 + ALPHA_ORIGINAL ** 2))

ESCALA_PROFUNDA = 0.15
ALPHA_PROFUNDA = 0.5

FATOR_LEAKY_PROFUNDA = (1.0 + ALPHA_PROFUNDA ** 2) / 2.0
FATOR_ATIVACAO_PROFUNDA = (
    ESCALA_PROFUNDA ** 2 * FATOR_LEAKY_PROFUNDA
)
GANHO_PROFUNDO = 1.0 / math.sqrt(FATOR_ATIVACAO_PROFUNDA)


def ativacao(x: torch.Tensor) -> torch.Tensor:
    if _N_CAMADAS_ATUAL > 16:
        return ESCALA_PROFUNDA * F.leaky_relu(
            x,
            negative_slope=ALPHA_PROFUNDA,
        )

    return F.leaky_relu(
        x,
        negative_slope=ALPHA_ORIGINAL,
    )


@torch.no_grad()
def inicializar(
    W: torch.Tensor,
    b: torch.Tensor,
    fan_in: int,
    fan_out: int,
    camada: int,
    n_camadas: int,
) -> None:
    global _N_CAMADAS_ATUAL

    _N_CAMADAS_ATUAL = n_camadas

    if n_camadas > 16:
        if camada == n_camadas:
            desvio = 0.5 / math.sqrt(fan_in)
        else:
            desvio = GANHO_PROFUNDO / math.sqrt(fan_in)
    else:
        if camada == n_camadas:
            desvio = 0.5 / math.sqrt(fan_in)
        else:
            fator_amortecimento = 1.0 - (0.0008 * n_camadas)
            desvio = (
                GANHO_BASE
                * fator_amortecimento
                / math.sqrt(fan_in)
            )

    W.normal_(mean=0.0, std=desvio)
    b.zero_()
