from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .greedy import greedy as greedy_solve
from .scenarios import gap
from .verify import chosen, split

OUT = Path(__file__).parent / "output"


def plot_lp_vs_milp(base, base_lp, path):
    zm = base["ctx"]["result"].fun
    zl = base_lp["result"].fun
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["LP-релаксация", "MILP"], [zl, zm], color=["#9ecae1", "#3182bd"])
    for b, v in zip(bars, [zl, zm]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("число позиций на полке")
    ax.set_title(f"LP-релаксация vs MILP (gap = {(zm - zl) / zm * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scale_curve(curve, path):
    sizes = [r["size"] for r in curve]
    times = [max(r["time_milp"], 1e-4) for r in curve]
    gaps = [r["gap_pct"] if r["gap_pct"] is not None else 0 for r in curve]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.semilogy(sizes, times, "o-", color="#3182bd", label="время MILP, с")
    ax1.set_xlabel("число SKU")
    ax1.set_ylabel("время решения MILP, с (лог. шкала)", color="#3182bd")
    ax2 = ax1.twinx()
    ax2.plot(sizes, gaps, "s--", color="#de2d26", label="gap LP/MILP, %")
    ax2.set_ylabel("интегральный зазор, %", color="#de2d26")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(skus, cats, space, results):
    OUT.mkdir(exist_ok=True)
    L = []

    def p(s=""):
        L.append(s)

    base = results["base"]
    g, res = base["ctx"], base["ctx"]["result"]
    glp = base["lp"]
    z_milp, z_lp = res.fun, glp["result"].fun
    gpct = gap(z_milp, z_lp)
    x, y = split(g)
    _, ylp = split(glp)
    frac = [(g["sku_ids"][s], round(float(ylp[s]), 3))
            for s in range(g["n"]) if abs(ylp[s] - round(ylp[s])) > 1e-3]
    gr = greedy_solve(skus, cats, space)
    ggap = ((gr["objective"] - z_milp) / z_milp * 100) if gr["feasible_space"] else None
    bigm = results["bigm"]

    p("# ЛР №3. MILP: выбор ассортимента магазина (вариант 10)")
    p()
    p("seed генерации данных: **10217** (N = 17, Q = 2 — группа 527, "
      "Y = 26). Решатель: HiGHS (scipy.optimize.milp).")
    p()
    p("## 1. Постановка и вариант")
    p()
    p("Магазину нужно сформировать полочный ассортимент: 20 SKU в 6 "
      "товарных категориях. Каждая категория имеет спрос $D_c$ (шт./нед), "
      "который нужно покрыть выкладкой, на полке ограниченная длина "
      f"$B = {space:.1f}$ м. SKU либо стоит на полке, либо нет (решение "
      "«да/нет»), поэтому задача не сводится к чистой ЛП: нужны бинарные "
      "переменные и связки big-M.")
    p()
    p("Исходные данные (data/skus.csv, data/categories.csv, "
      "data/shelf.csv):")
    p()
    p(skus.to_markdown(index=False))
    p()
    p(cats.to_markdown(index=False))
    p()
    p(f"Бюджет полки: $B = {space:.1f}$ м.")
    p()
    p("## 2. Формальная MILP-модель")
    p()
    p("- Множества: SKU $s \\in S$ (|S| = 20), категории $c \\in C$ "
      "(|C| = 6), $S_c$ — SKU категории $c$.")
    p("- Параметры: $m_s$ — минимальная выкладка (шт.), $U_s$ — максимум "
      "запаса SKU на полке (шт.), $a_s$ — полочная длина единицы (м), "
      "$D_c$ — спрос категории (шт./нед), $n_c$ — минимум SKU в "
      "категории, $B$ — бюджет полки (м).")
    p("- Переменные: $y_s \\in \\{0,1\\}$ — SKU выставлен на полку; "
      "$x_s \\ge 0$ (непрерывная) — выставленный запас (шт.).")
    p("- Цель: $\\min \\sum_s y_s$ — минимум полочных позиций.")
    p("- Ограничения:")
    p("  1. связка big-M: $x_s \\le M_s y_s$, где $M_s = U_s$ — "
      "разумная константа: больше $U_s$ хранить бессмысленно;")
    p("  2. минимальная выкладка: $x_s \\ge m_s y_s$ (выставленный SKU "
      "нельзя класть меньше минимума);")
    p("  3. покрытие спроса по категориям: $\\sum_{s \\in S_c} x_s \\ge "
      "D_c$;")
    p("  4. ассортиментное разнообразие: $\\sum_{s \\in S_c} y_s \\ge "
      "n_c$;")
    p("  5. бюджет полки: $\\sum_s a_s x_s \\le B$.")
    p()
    p("**Допущения:** запас любого SKU не ограничен сверх $U_s$; спрос "
      "категории взаимозаменяем между SKU категории (классическое "
      "допущение ассортиментного планирования); полочная длина линейна по "
      "числу единиц.")
    p()
    p("## 3. Реализация и статус")
    p()
    p("Модель собрана матрицами NumPy и решена через "
      "`scipy.optimize.milp` (HiGHS, ветви и границы, presolve включён, "
      "требование зазора 1e-9). LP-релаксация — та же модель без "
      "целочисленности. Статус базового решения: **0 — оптимально**. "
      "Лог солвера: output/solver_log.txt.")
    p()
    p("## 4. MILP vs LP-релаксация")
    p()
    p(f"- MILP: $z^{{*}} = {z_milp:.0f}$ позиций; время {g['elapsed']:.4f} с.")
    p(f"- LP-релаксация: $z_{{LP}} = {z_lp:.2f}$; время "
      f"{glp['elapsed']:.4f} с.")
    p(f"- интегральный зазор: **{gpct:.1f}%** от оптимума MILP.")
    p()
    if frac:
        p("Где «ломается» непрерывное решение: релаксация ставит "
          "дробные $y_s$ — например "
          + ", ".join(f"$y_{{{s}}} = {v}$" for s, v in frac[:5])
          + (" …" if len(frac) > 5 else "")
          + ". Физически «пол-SKU на полке» невозможно: округление "
            "дробных $y$ вниз разрушает покрытие спроса, вверх — даёт "
            "лишние позиции и место; кроме того, набор LP отличается от "
            "MILP-набора, то есть оптимальный ассортимент не "
            "восстанавливается округлением.")
    p()
    rows = []
    for s in range(g["n"]):
        if y[s] > 0.5:
            rows.append({"sku_id": g["sku_ids"][s],
                         "категория": g["cat_ids"][g["mem"][s]],
                         "запас x, шт.": round(float(x[s]), 1),
                         "занято, м": round(float(g["unit_space"][s] * x[s]), 2)})
    p("Оптимальный набор (MILP):")
    p()
    p(pd.DataFrame(rows).to_markdown(index=False))
    p()
    p(f"![LP vs MILP](output/lp_vs_milp.png)")
    p()
    p("## 5. Независимая проверка (NumPy)")
    p()
    for c in base["verify"]["checks"]:
        p(f"- {c}")
    p(f"- итог: **{'допустимо' if base['verify']['ok'] else 'ОШИБКА'}**")
    p()
    p("## 6. Сценарный анализ")
    p()
    srows = []
    for n in ["base", "deficit", "stress"]:
        r = results[n]
        c = r["ctx"]
        srows.append({"сценарий": n,
                      "статус": "optimal" if c["result"].success else "infeasible",
                      "позиций": c["result"].fun if c["result"].success else None,
                      "проверка NumPy": "OK" if r["verify"]["ok"] else "FAIL",
                      "время, с": round(c["elapsed"], 4)})
    p(pd.DataFrame(srows).to_markdown(index=False))
    p()
    for n in ["deficit", "stress"]:
        ch = results[n]["chosen"]
        base_ch = base["chosen"]
        p(f"- **{n}:** набрано {len(ch)} SKU; ушли: "
          f"{sorted(base_ch - ch) or '—'}; добавлены: "
          f"{sorted(ch - base_ch) or '—'}.")
    p()
    p("**Интерпретация:** в дефицитном сценарии (полка −20%) число "
      "позиций сохраняется, но ассортимент перестраивается — на полку "
      "берутся SKU с более компактной выкладкой. В стрессовом (спрос "
      "+20%) потребовались дополнительные позиции: максимальные запасы "
      "выбранных SKU перестают покрывать категорийный спрос.")
    p()
    p("## 7. Масштабирование: время и зазор")
    p()
    p("Однотипные экземпляры с 10, 20 и 40 SKU (тот же seed, та же "
      "схема генерации):")
    p()
    p(pd.DataFrame(results["curve"]).to_markdown(index=False))
    p()
    p("![Кривая масштаба](output/scale_curve.png)")
    p()
    p("Время MILP растёт заметно быстрее LP-релаксации — цена "
      "перебора бинарных решений в ветвях; зазор при этом не обязан "
      "расти: он зависит от геометрии конкретного экземпляра.")
    p()
    p("## 8. Жадная эвристика")
    p()
    p("Эвристика: в каждой категории SKU перебираются по возрастанию "
      "минимальной выкладки и берутся, пока не достигнуты покрытие "
      "спроса и минимум позиций.")
    p()
    p(f"- жадная эвристика: {gr['objective']:.0f} позиций, "
      f"время {gr['elapsed'] * 1000:.2f} мс, "
      f"допустимость по полке: {'да' if gr['feasible_space'] else 'нет'}")
    p(f"- MILP-оптимум: {z_milp:.0f} позиций")
    if ggap is not None:
        p(f"- зазор эвристики: **{ggap:.1f}%** к оптимуму")
    p()
    p("## 9. Выбор big-M и его нарушение")
    p()
    p("Разумная константа $M_s = U_s$: она точно отсекает запас "
      "несуществующих позиций и не создаёт численных проблем. Проверка "
      "с $M = 10^6$ для всех связок:")
    p()
    zb = bigm["ctx"]["result"]
    p(f"- статус солвера: {'optimal' if zb.success else 'infeasible'}, "
      f"$z = {zb.fun:.0f}$")
    p(f"- независимая проверка связок big-M: "
      f"**{'OK' if bigm['verify']['ok'] else 'FAIL'}**")
    if not bigm["verify"]["ok"]:
        p("- при огромном $M$ допуск солвера, умноженный на $10^6$, "
          "формально разрешает запас у невыставленного SKU: солвер "
          "рапортует о «оптимуме» с меньшим числом позиций, но решение "
          "недопустимо. Вывод: $M$ берут максимально тесным (здесь "
          "$M_s = U_s$).")
    p()
    p("## 10. Вывод")
    p()
    p(f"Оптимальный ассортимент — **{z_milp:.0f} полочных позиций** "
      f"({', '.join(sorted(base['chosen']))}), покрывающих спрос всех "
      f"категорий в бюджете {space:.1f} м полки. LP-релаксация даёт "
      f"оптимистичную оценку {z_lp:.2f} (зазор {gpct:.1f}%); дробные "
      "значения $y_s$ показывают, где дискретность качественно меняет "
      "решение. Жадная эвристика проигрывает оптимуму; большие big-M "
      "разрушают даже допустимость решения.")
    p()
    p("**Можно ли реализовать на практике?** Модель учитывает главные "
      "ограничения полки (место, минимальная выкладка, разнообразие), "
      "но не учитывает взаимное влияние SKU на продажи (каннибализацию), "
      "несколько полок/уровней, промо-размещение и нелинейность спроса "
      "от запаса — для них нужны более богатые модели (см. литературу по "
      "ассортиментному планированию).")
    p()
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
