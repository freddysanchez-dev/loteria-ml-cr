# app/ui.py

def formatear_top(top_numeros):
    """Convierte [(num, prob), ...] a listas separadas para el template."""
    numeros = [n for n, _ in top_numeros]
    probs = [f"{p*100:.2f}%" for _, p in top_numeros]
    return numeros, probs


def construir_contexto(top_numeros):
    """Prepara el dict que consumirá index.html."""
    numeros, probs = formatear_top(top_numeros)
    numero_recomendado = f"{numeros[0]:02d}" if numeros else "--"
    prob_recomendado = probs[0] if probs else "0.00%"
    return {
        'numero_recomendado': numero_recomendado,
        'prob_recomendado': prob_recomendado,
        'top_numeros': numeros,
        'top_probs': probs,
    }