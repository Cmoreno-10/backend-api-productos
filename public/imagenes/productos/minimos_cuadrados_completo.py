# ============================================================
# MÍNIMOS CUADRADOS - GUÍA COMPLETA CON MATRICES Y VECTORES
# Para ejecutar en Google Colab: cada sección es independiente
# ============================================================

# ─── INSTALACIÓN (en Colab ejecutar esta celda primero) ───
# !pip install numpy pandas matplotlib scikit-learn scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
np.random.seed(42)

# ===========================================================
# PREGUNTA 1 y 2: PAPEL DE MATRICES + MATRIZ DE DISEÑO
# ===========================================================
print("=" * 60)
print("PARTE 1: Modelo matricial  y = Xβ + ε")
print("=" * 60)

# Simulamos datos de casas: precio en función de m² y habitaciones
n = 50  # número de observaciones
metros2      = np.random.uniform(50, 200, n)
habitaciones = np.random.randint(1, 6, n).astype(float)
ruido        = np.random.normal(0, 15, n)

# Variable objetivo: precio en miles de USD
precio = 2.5 * metros2 + 10 * habitaciones + 30 + ruido

# ── Construcción de la Matriz de Diseño X ──
# La primera columna es siempre de unos → intercepto β₀
# Luego vienen las variables predictoras
X = np.column_stack([
    np.ones(n),      # columna de 1s para el intercepto
    metros2,         # variable 1
    habitaciones     # variable 2
])

y = precio

print(f"\nForma de X (matriz de diseño): {X.shape}  → {n} observaciones, 3 coeficientes")
print(f"Forma de y (variable objetivo): {y.shape}")
print(f"\nPrimeras 5 filas de X:\n{X[:5]}")
print(f"\nPrimeras 5 valores de y:\n{y[:5].round(2)}")

# ===========================================================
# PREGUNTA 2: Solución de mínimos cuadrados
# Ecuación normal: β = (XᵀX)⁻¹ Xᵀy
# ===========================================================
print("\n" + "=" * 60)
print("PARTE 2: Ecuación Normal  β = (XᵀX)⁻¹ Xᵀy")
print("=" * 60)

# Paso a paso de la ecuación normal
XtX   = X.T @ X          # Producto XᵀX  → (3×50)(50×3) = 3×3
XtX_inv = np.linalg.inv(XtX)  # Inversa de XᵀX
Xty   = X.T @ y          # Producto Xᵀy  → (3×50)(50×1) = 3×1
beta  = XtX_inv @ Xty    # Coeficientes estimados → β

print(f"\nMatriz XᵀX (3×3):\n{XtX.round(2)}")
print(f"\n(XᵀX)⁻¹:\n{XtX_inv.round(6)}")
print(f"\nVector Xᵀy: {Xty.round(2)}")
print(f"\nCoeficientes estimados β:")
print(f"  β₀ (intercepto) = {beta[0]:.3f}")
print(f"  β₁ (metros²)    = {beta[1]:.3f}  ← precio por m²")
print(f"  β₂ (habitac.)   = {beta[2]:.3f}  ← precio por habitación")
print(f"\nValores reales usados: β₀=30, β₁=2.5, β₂=10")

# Predicciones y residuos
y_pred = X @ beta
residuos = y - y_pred
print(f"\nRMSE manual: {np.sqrt(np.mean(residuos**2)):.4f}")

# ===========================================================
# PREGUNTA 3: ORTOGONALIDAD Y PROYECCIÓN
# Verificar que Xᵀε ≈ 0 (residuos ⊥ al espacio columna)
# ===========================================================
print("\n" + "=" * 60)
print("PARTE 3: Verificación de Ortogonalidad  Xᵀε ≈ 0")
print("=" * 60)

Xt_eps = X.T @ residuos  # Debe ser vector de ceros
print(f"\nXᵀε = {Xt_eps.round(8)}")
print("→ Efectivamente ≈ 0. Los residuos son perpendiculares a Col(X)")

# Visualizar proyección en 2D (solo metros² para simplicidad)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica 1: Proyección ortogonal
ax = axes[0]
ax.scatter(metros2, precio, color='steelblue', alpha=0.6, label='Datos reales (y)', s=50)
ax.plot(np.sort(metros2),
        beta[0] + beta[1]*np.sort(metros2) + beta[2]*3,
        color='crimson', linewidth=2, label='ŷ = Xβ (proyección)')

# Dibujar algunos residuos como segmentos verticales
for i in range(0, n, 5):
    ax.plot([metros2[i], metros2[i]],
            [y_pred[i], precio[i]],
            color='orange', alpha=0.5, linewidth=1)

ax.set_xlabel('Metros cuadrados')
ax.set_ylabel('Precio (miles USD)')
ax.set_title('Proyección ortogonal: y sobre Col(X)\n(segmentos naranja = residuos ε)')
ax.legend()
ax.grid(True, alpha=0.3)

# Gráfica 2: Residuos vs predichos (verificar ortogonalidad)
ax2 = axes[1]
ax2.scatter(y_pred, residuos, color='darkorange', alpha=0.6, s=50)
ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')
ax2.set_xlabel('Valores predichos ŷ')
ax2.set_ylabel('Residuos ε = y − ŷ')
ax2.set_title('Residuos vs Predichos\n(deben estar centrados en 0)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('proyeccion_ortogonal.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ Gráfica guardada: proyeccion_ortogonal.png")

# ===========================================================
# PREGUNTA 4: SIMETRÍA Y POSITIVIDAD DEFINIDA DE XᵀX
# ===========================================================
print("\n" + "=" * 60)
print("PARTE 4: Propiedades de XᵀX")
print("=" * 60)

# Verificar simetría: XᵀX = (XᵀX)ᵀ
es_simetrica = np.allclose(XtX, XtX.T)
print(f"\n¿XᵀX es simétrica? {es_simetrica}  → (XᵀX)ᵀ = XᵀX")

# Verificar positividad definida mediante valores propios (eigenvalues)
eigenvalores = np.linalg.eigvals(XtX)
print(f"\nValores propios de XᵀX: {eigenvalores.round(2)}")
print(f"¿Todos positivos? {np.all(eigenvalores > 0)}")
print("→ Si todos los eigenvalores > 0, XᵀX es positiva definida")
print("→ Esto garantiza que la solución de mínimos cuadrados es ÚNICA")

# Número de condición: mide qué tan invertible es XᵀX
cond = np.linalg.cond(XtX)
print(f"\nNúmero de condición de XᵀX: {cond:.2f}")
print("→ Si este número es muy grande (>10⁶), hay multicolinealidad")

# Visualizar XᵀX como heatmap
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(XtX, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_title('Matriz XᵀX\n(simétrica y positiva definida)')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Intercepto', 'Metros²', 'Habitac.'])
ax.set_yticklabels(['Intercepto', 'Metros²', 'Habitac.'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{XtX[i,j]:.0f}', ha='center', va='center',
                color='black', fontsize=9)
plt.tight_layout()
plt.savefig('XtX_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ===========================================================
# PREGUNTA 5: EXTENSIONES - WLS Y GLS
# ===========================================================
print("\n" + "=" * 60)
print("PARTE 5: Mínimos cuadrados ponderados (WLS) y generalizados (GLS)")
print("=" * 60)

# ── WLS: cuando la varianza del error no es constante ──
# Ejemplo: los datos más nuevos son más confiables → más peso
pesos = np.linspace(0.5, 2.0, n)  # pesos crecientes (observaciones recientes)

# Matriz de pesos W (diagonal)
W = np.diag(pesos)

# Ecuación WLS: β = (XᵀWX)⁻¹ XᵀWy
XtWX  = X.T @ W @ X
beta_wls = np.linalg.inv(XtWX) @ (X.T @ W @ y)

print(f"\nWLS - Coeficientes con pesos:")
print(f"  β₀ = {beta_wls[0]:.3f}  (OLS fue {beta[0]:.3f})")
print(f"  β₁ = {beta_wls[1]:.3f}  (OLS fue {beta[1]:.3f})")
print(f"  β₂ = {beta_wls[2]:.3f}  (OLS fue {beta[2]:.3f})")

# ── GLS: cuando los errores tienen correlación (ej: series de tiempo) ──
# Simulamos errores con correlación AR(1)
rho = 0.6  # correlación entre errores consecutivos
Omega = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        Omega[i, j] = rho ** abs(i - j)  # decae con la distancia

# GLS: β = (XᵀΩ⁻¹X)⁻¹ XᵀΩ⁻¹y
Omega_inv = np.linalg.inv(Omega)
beta_gls = np.linalg.inv(X.T @ Omega_inv @ X) @ (X.T @ Omega_inv @ y)

print(f"\nGLS - Coeficientes con errores correlacionados (ρ={rho}):")
print(f"  β₀ = {beta_gls[0]:.3f}")
print(f"  β₁ = {beta_gls[1]:.3f}")
print(f"  β₂ = {beta_gls[2]:.3f}")

# Comparación visual
fig, ax = plt.subplots(figsize=(8, 4))
metodos = ['OLS\n(estándar)', 'WLS\n(ponderado)', 'GLS\n(generalizado)']
betas_comp = np.array([beta, beta_wls, beta_gls])
x_pos = np.arange(3)
width = 0.25
ax.bar(x_pos - width, betas_comp[:, 0], width, label='β₀ (intercepto)', color='steelblue')
ax.bar(x_pos,         betas_comp[:, 1], width, label='β₁ (metros²)', color='darkorange')
ax.bar(x_pos + width, betas_comp[:, 2], width, label='β₂ (habitaciones)', color='green')
ax.set_xticks(x_pos)
ax.set_xticklabels(metodos)
ax.set_ylabel('Valor del coeficiente')
ax.set_title('Comparación: OLS vs WLS vs GLS')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('comparacion_ols_wls_gls.png', dpi=150, bbox_inches='tight')
plt.show()

# ===========================================================
# PREGUNTAS 6, 7, 8: APLICACIÓN REAL Y ESTRUCTURA DE DATOS
# ===========================================================
print("\n" + "=" * 60)
print("PARTE 6-8: Base de datos simulada - Predicción de salarios")
print("=" * 60)

# Simulamos base de datos de RRHH para predecir salario
np.random.seed(123)
m = 200  # empleados

datos = pd.DataFrame({
    'años_experiencia':  np.random.randint(0, 30, m),
    'nivel_educacion':   np.random.choice([1, 2, 3, 4], m),  # 1=bach, 4=doctorado
    'horas_semana':      np.random.uniform(35, 60, m),
    'certificaciones':   np.random.randint(0, 5, m),
    'sector':            np.random.choice([0, 1], m),  # 0=público, 1=privado
})

# Salario en miles USD/año
datos['salario'] = (
    25 +
    1.8 * datos['años_experiencia'] +
    8   * datos['nivel_educacion'] +
    0.3 * datos['horas_semana'] +
    3   * datos['certificaciones'] +
    12  * datos['sector'] +
    np.random.normal(0, 8, m)
)

print(f"\nBase de datos de RRHH:")
print(datos.head(8).to_string(index=False))
print(f"\nEstadísticas descriptivas:")
print(datos.describe().round(2))

# Construcción de la matriz de diseño desde el DataFrame
print(f"\n── Estructura de la Matriz de Diseño ──")
variables = ['años_experiencia', 'nivel_educacion', 'horas_semana',
             'certificaciones', 'sector']
X_rrhh = np.column_stack([np.ones(m), datos[variables].values])
y_rrhh = datos['salario'].values

print(f"Forma de X: {X_rrhh.shape}  ({m} empleados × 6 coeficientes)")
print(f"Columnas: [intercepto] + {variables}")

# ===========================================================
# PREGUNTA 9: INTERPRETACIÓN DE RESULTADOS
# ===========================================================
print("\n" + "=" * 60)
print("PARTE 9: Interpretación de coeficientes")
print("=" * 60)

beta_rrhh = np.linalg.inv(X_rrhh.T @ X_rrhh) @ (X_rrhh.T @ y_rrhh)

nombres_coef = ['Intercepto (base)', 'Años experiencia', 'Nivel educación',
                'Horas/semana', 'Certificaciones', 'Sector privado']

print("\nCoeficientes estimados e interpretación:")
for nombre, coef in zip(nombres_coef, beta_rrhh):
    print(f"  {nombre:25s}: {coef:8.3f} miles USD")

print("\nInterpretación económica:")
print(f"  → Cada año de experiencia suma ${beta_rrhh[1]*1000:.0f} al salario anual")
print(f"  → Cada nivel educativo suma ${beta_rrhh[2]*1000:.0f} al salario")
print(f"  → Trabajar en sector privado suma ${beta_rrhh[5]*1000:.0f} al salario")

# ===========================================================
# PREGUNTA 10: TÉCNICAS DE VALIDACIÓN
# ===========================================================
print("\n" + "=" * 60)
print("PARTE 10: Validación del modelo")
print("=" * 60)

# Usando sklearn para validación cruzada
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

X_sk = datos[variables]
y_sk = datos['salario']

modelo = LinearRegression()
modelo.fit(X_sk, y_sk)
y_pred_rrhh = modelo.predict(X_sk)

# Métricas básicas
r2  = r2_score(y_sk, y_pred_rrhh)
rmse = np.sqrt(mean_squared_error(y_sk, y_pred_rrhh))
residuos_rrhh = y_sk - y_pred_rrhh

print(f"\nMétricas de ajuste (sobre el conjunto de entrenamiento):")
print(f"  R²   = {r2:.4f}  → El modelo explica el {r2*100:.1f}% de la varianza")
print(f"  RMSE = {rmse:.4f} miles USD")

# Validación cruzada K-Fold (la técnica más robusta)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_r2   = cross_val_score(modelo, X_sk, y_sk, cv=kf, scoring='r2')
cv_scores_rmse = np.sqrt(-cross_val_score(modelo, X_sk, y_sk, cv=kf,
                                          scoring='neg_mean_squared_error'))

print(f"\nValidación cruzada 5-Fold:")
print(f"  R²   por fold: {cv_scores_r2.round(4)}")
print(f"  R²   promedio: {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
print(f"  RMSE promedio: {cv_scores_rmse.mean():.4f} ± {cv_scores_rmse.std():.4f} miles USD")

# Test de normalidad de residuos (Shapiro-Wilk)
stat, p_valor = stats.shapiro(residuos_rrhh)
print(f"\nTest Shapiro-Wilk (normalidad de residuos):")
print(f"  Estadístico = {stat:.4f}, p-valor = {p_valor:.4f}")
print(f"  Residuos {'siguen' if p_valor > 0.05 else 'NO siguen'} distribución normal (α=0.05)")

# Dashboard final de validación
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# 1. Real vs Predicho
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_sk, y_pred_rrhh, alpha=0.5, color='steelblue', s=30)
lims = [min(y_sk.min(), y_pred_rrhh.min()), max(y_sk.max(), y_pred_rrhh.max())]
ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Predicción perfecta')
ax1.set_xlabel('Salario real')
ax1.set_ylabel('Salario predicho')
ax1.set_title(f'Real vs Predicho\nR² = {r2:.3f}')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Residuos vs predichos
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred_rrhh, residuos_rrhh, alpha=0.5, color='darkorange', s=30)
ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')
ax2.set_xlabel('Valores predichos')
ax2.set_ylabel('Residuos')
ax2.set_title('Residuos vs Predichos\n(¿patrón aleatorio?)')
ax2.grid(True, alpha=0.3)

# 3. Histograma de residuos
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuos_rrhh, bins=25, color='green', alpha=0.7, edgecolor='darkgreen')
ax3.set_xlabel('Residuos')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Distribución de residuos\n(debe ser ≈ normal)')
ax3.grid(True, alpha=0.3)

# 4. Q-Q plot
ax4 = fig.add_subplot(gs[1, 0])
stats.probplot(residuos_rrhh, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot de residuos\n(puntos sobre la línea = normalidad)')
ax4.grid(True, alpha=0.3)

# 5. Importancia de variables (coeficientes estandarizados)
ax5 = fig.add_subplot(gs[1, 1])
X_std = (datos[variables] - datos[variables].mean()) / datos[variables].std()
modelo_std = LinearRegression().fit(X_std, y_sk)
coefs_std  = modelo_std.coef_
colores    = ['green' if c > 0 else 'red' for c in coefs_std]
ax5.barh(variables, coefs_std, color=colores, alpha=0.7)
ax5.axvline(0, color='black', linewidth=1)
ax5.set_xlabel('Coeficiente estandarizado')
ax5.set_title('Importancia de variables\n(coeficientes β estandarizados)')
ax5.grid(True, alpha=0.3)

# 6. R² por fold (validación cruzada)
ax6 = fig.add_subplot(gs[1, 2])
folds = [f'Fold {i+1}' for i in range(5)]
ax6.bar(folds, cv_scores_r2, color='purple', alpha=0.7)
ax6.axhline(cv_scores_r2.mean(), color='red', linewidth=2,
            linestyle='--', label=f'Promedio R²={cv_scores_r2.mean():.3f}')
ax6.set_ylabel('R²')
ax6.set_title('Validación cruzada 5-Fold\n(consistencia del modelo)')
ax6.set_ylim(0, 1)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('Dashboard de Validación - Modelo de Mínimos Cuadrados\nPredicción de salarios (datos simulados RRHH)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('dashboard_validacion_mincuadrados.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n→ Dashboard guardado: dashboard_validacion_mincuadrados.png")

# ===========================================================
# RESUMEN EJECUTIVO DE COEFICIENTES PARA TOMA DE DECISIONES
# ===========================================================
print("\n" + "=" * 60)
print("RESUMEN: Toma de decisiones basada en los coeficientes")
print("=" * 60)

for var, coef in zip(variables, modelo.coef_):
    print(f"  +1 {var:25s} → salario {'+'if coef>0 else ''}{coef:.2f}k USD/año")

print(f"\nConclusion práctica:")
print(f"  La inversión más rentable en capital humano es el sector privado ({modelo.coef_[-1]:.1f}k)")
print(f"  seguido de nivel educativo ({modelo.coef_[1]:.1f}k por nivel) y experiencia ({modelo.coef_[0]:.1f}k/año)")
print(f"\nR² validación cruzada = {cv_scores_r2.mean():.3f}")
print("El modelo es robusto y generaliza bien a nuevos datos.")
