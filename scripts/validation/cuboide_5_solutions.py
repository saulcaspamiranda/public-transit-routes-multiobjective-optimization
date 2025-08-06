import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Mostrar en navegador
pio.renderers.default = 'browser'

# === Paso 1: Datos fijos (las 5 soluciones) ===
data = {
    'f1': [-3655, -4030, -5410, -3684, -4120],
    'f2': [153769, 160126, 205959, 154194, 162932],
    'f3': [-247854, -255098, -287624, -249005, -262614]
}
df = pd.DataFrame(data)

# === Paso 2: Inicializar crowding distance ===
df['crowding_distance'] = 0.0
objectives = ['f1', 'f2', 'f3']

for obj in objectives:
    sorted_idx = df[obj].sort_values(ascending=False).index.tolist()
    f_max, f_min = df[obj].max(), df[obj].min()
    
    # Extremos
    df.loc[sorted_idx[0], 'crowding_distance'] = np.inf
    df.loc[sorted_idx[-1], 'crowding_distance'] = np.inf
    
    if f_max - f_min == 0:
        continue

    for j in range(1, len(df) - 1):
        i = sorted_idx[j]
        prev_val = df.loc[sorted_idx[j - 1], obj]
        next_val = df.loc[sorted_idx[j + 1], obj]
        df.loc[i, 'crowding_distance'] += (next_val - prev_val) / (f_max - f_min)

# === Paso 3: Graficar con plotly ===
fig = go.Figure()

# Puntos
fig.add_trace(go.Scatter3d(
    x=df['f1'], y=df['f2'], z=df['f3'],
    mode='markers+text',
    text=[f'Sol {i}' for i in df.index],
    marker=dict(
        size=6,
        color=df['crowding_distance'],
        colorscale='Viridis',
        colorbar=dict(title='Crowding')
    ),
    name='Soluciones'
))

# Cuboides para internas
for i in df.index:
    if not np.isfinite(df.loc[i, 'crowding_distance']):
        continue

    # Calcular span (ancho) por eje
    spans = {}
    for obj in objectives:
        sorted_idx = df[obj].sort_values(ascending=False).index.tolist()
        j = sorted_idx.index(i)
        if j == 0 or j == len(df) - 1:
            spans[obj] = None
            continue
        prev_val = df.loc[sorted_idx[j - 1], obj]
        next_val = df.loc[sorted_idx[j + 1], obj]
        spans[obj] = abs(next_val - prev_val)

    # Validar
    if None in spans.values():
        continue

    # Centro del cuboide
    x0, y0, z0 = df.loc[i, ['f1', 'f2', 'f3']]
    dx, dy, dz = spans['f1'], spans['f2'], spans['f3']

    x = [x0 - dx/2, x0 + dx/2]
    y = [y0 - dy/2, y0 + dy/2]
    z = [z0 - dz/2, z0 + dz/2]

    # Vértices
    vertices = {
        'x': [x[0], x[1], x[1], x[0], x[0], x[1], x[1], x[0]],
        'y': [y[0], y[0], y[1], y[1], y[0], y[0], y[1], y[1]],
        'z': [z[0], z[0], z[0], z[0], z[1], z[1], z[1], z[1]]
    }

    # Caras (12 triángulos para 6 caras del cuboide)
    faces = dict(
        i=[0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 6, 6],
        j=[1, 3, 2, 5, 3, 6, 5, 7, 6, 1, 7, 2],
        k=[2, 1, 5, 2, 6, 3, 6, 3, 1, 0, 3, 7]
    )

    fig.add_trace(go.Mesh3d(
        x=vertices['x'],
        y=vertices['y'],
        z=vertices['z'],
        i=faces['i'],
        j=faces['j'],
        k=faces['k'],
        opacity=0.2,
        color='gray',
        name=f'Cuboide {i}',
        showscale=False
    ))

# Configurar ejes y título
fig.update_layout(
    title='Visualización de Crowd Distance y Cuboides (Deb 2001)',
    scene=dict(
        xaxis_title='f1: -Conexión de áreas',
        yaxis_title='f2:  Tiempo total de viaje',
        zaxis_title='f3: -Concurrencia servida'
    )
)

fig.show()

