from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from .greedy import greedy
from .model import split
from .scenarios import bottlenecks

OUT = Path(__file__).parent / "output"


def plot_network(warehouses, shops, routes, g, path):
    res = g["result"]
    x, _ = split(g)
    G = nx.DiGraph()
    pos = {}
    for i, wid in enumerate(g["wids"]):
        G.add_node(wid, kind="w")
        pos[wid] = (0, -i)
    for j, sid in enumerate(g["sids"]):
        G.add_node(sid, kind="m")
        pos[sid] = (1, -j)
    fig, ax = plt.subplots(figsize=(10, 7))
    for t, (a, b) in enumerate(g["arc_ij"]):
        if x[t] <= 1e-9:
            continue
        sat = g["cap_used"][t] > 0 and x[t] >= g["cap_used"][t] - 1e-6
        G.add_edge(a, b, flow=x[t], cap=g["cap_used"][t], sat=sat)
    nx.draw_networkx_nodes(G, pos, nodelist=g["wids"], node_color="#9ecae1",
                           node_shape="s", node_size=900, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=g["sids"], node_color="#fdae6b",
                           node_shape="o", node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
    plain = [(a, b) for a, b, d in G.edges(data=True) if not d["sat"]]
    sat_e = [(a, b) for a, b, d in G.edges(data=True) if d["sat"]]
    nx.draw_networkx_edges(G, pos, edgelist=plain, edge_color="#999999",
                           arrows=True, ax=ax, node_size=900)
    nx.draw_networkx_edges(G, pos, edgelist=sat_e, edge_color="#de2d26",
                           width=2.5, arrows=True, ax=ax, node_size=900)
    labels = {(a, b): f'{d["flow"]:.1f}/{d["cap"]:.0f}' for a, b, d in
              G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7,
                                 label_pos=0.72, ax=ax,
                                 bbox=dict(fc="white", ec="none", alpha=0.7,
                                           pad=0.5))
    ax.set_title("Схема сети: поток/пропускная способность (красным — "
                 "насыщенные дуги)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(results, path):
    names = list(results)
    objs = [results[n]["ctx"]["result"].fun for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, objs, color="#3182bd")
    ax.set_ylabel("цель, руб.")
    ax.set_title("Целевая функция по сценариям")
    for i, v in enumerate(objs):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(warehouses, shops, routes, results):
    OUT.mkdir(exist_ok=True)
    L = []

    def p(s=""):
        L.append(s)

    base = results["base"]
    g = base["ctx"]
    res = g["result"]
    x, u = split(g)
    gr = greedy(warehouses, shops, routes)
    gap = (gr["objective"] - res.fun) / abs(res.fun) * 100

    p("# ЛР №2. Транспортная задача с дефицитом и штрафами (вариант 10)")
    p()
    p("seed генерации данных: **10217** (N = 17, Q = 2 — группа 527, "
      "Y = 26). Решатель: HiGHS (scipy.optimize.linprog, method='highs').")
    p()
    p("## 1. Постановка и сеть")
    p()
    p("Региональная сеть снабжения: 4 склада (предложение, шт.) и 6 "
      "магазинов (спрос, шт.). Суммарный запас меньше суммарного спроса, "
      "поэтому вводится переменная дефицита $u_j$ — неудовлетворённый "
      "спрос магазина $j$; за каждую недопоставленную единицу начисляется "
      "штраф $p_j$ (потеря выручки/клиентов). Каждый маршрут имеет "
      "пропускную способность (вместимость рейсов).")
    p()
    p("Исходные данные (data/warehouses.csv, data/shops.csv, "
      "data/routes.csv):")
    p()
    p(warehouses.to_markdown(index=False))
    p()
    p(shops.to_markdown(index=False))
    p()
    p("Маршрутов: 24 (полный двудольный граф); стоимость перевозки — от "
      f"{g['cost'].min():.2f} до {g['cost'].max():.2f} руб./шт., "
      f"пропускная способность маршрута — от {int(g['cap'].min())} до "
      f"{int(g['cap'].max())} шт.")
    p()
    p("## 2. Формальная модель (ЛП)")
    p()
    p("- Множества: склады $i \\in W$ (|W| = 4), магазины $j \\in M$ "
      "(|M| = 6), маршруты (дуги) $(i,j) \\in A$ (|A| = 24).")
    p("- Переменные: $x_{ij} \\ge 0$ — перевезённый объём по маршруту "
      "$(i,j)$ (шт.); $u_j \\ge 0$ — неудовлетворённый спрос магазина "
      "$j$ (шт.).")
    p("- Параметры: $s_i$ — запас склада (шт.), $d_j$ — спрос (шт.), "
      "$c_{ij}$ — тариф (руб./шт.), $q_{ij}$ — пропускная способность "
      "маршрута (шт.), $p_j$ — штраф за дефицит (руб./шт.).")
    p("- Цель: $\\min z = \\sum_{(i,j)} c_{ij} x_{ij} + \\sum_j p_j "
      "u_j$ (руб.).")
    p("- Ограничения: отгрузка со склада $\\sum_j x_{ij} \\le s_i$ (шт.); "
      "баланс магазина $\\sum_i x_{ij} + u_j \\ge d_j$ (шт.); границы дуг "
      "$0 \\le x_{ij} \\le q_{ij}$ (шт.); $0 \\le u_j \\le d_j$.")
    p()
    p("**Допущения:** груз делимый (усреднённая постановка), тарифы "
      "линейны, время в пути не моделируется (учтено в тарифе), штраф "
      "линейно зависит от недопоставки.")
    p()
    p("## 3. Реализация и статус решения")
    p()
    p("Модель построена в SciPy (30 переменных: 24 потока + 6 дефицитов; "
      "34 ограничения), решатель HiGHS (двойственный симплекс, presolve "
      "включён). Статус: **0 — оптимально**; итераций: "
      f"{res.nit}; время: {g['elapsed']:.4f} с. Лог солвера: "
      "output/solver_log.txt.")
    p()
    p(f"**Оптимум: $z^{{*}} = {res.fun:.2f}$ руб.** "
      f"(перевозки {g['cost'] @ x:.2f} руб. + штрафы "
      f"{g['penalty'] @ u:.2f} руб.).")
    p()
    p("## 4. Найденное решение")
    p()
    rows = []
    for t, (a, b) in enumerate(g["arc_ij"]):
        if x[t] > 1e-9:
            rows.append({"маршрут": f"{a}→{b}", "поток, шт.": round(float(x[t]), 2),
                         "лимит, шт.": g["cap_used"][t],
                         "тариф, руб./шт.": g["cost"][t],
                         "стоимость, руб.": round(float(x[t] * g["cost"][t]), 2)})
    p(pd.DataFrame(rows).to_markdown(index=False))
    p()
    drows = [{"магазин": sid, "поставлено, шт.": round(float(sum(
        x[t] for t, (_, bb) in enumerate(g["arc_ij"]) if bb == sid)), 2),
        "дефицит $u_j$, шт.": round(float(u[j]), 2),
        "спрос, шт.": g["demand_used"][j],
        "штраф, руб.": round(float(u[j] * g["penalty"][j]), 2)}
        for j, sid in enumerate(g["sids"])]
    p(pd.DataFrame(drows).to_markdown(index=False))
    p()
    p("![Схема сети с потоками](output/network_flow.png)")
    p()
    p("## 5. Независимая проверка (NumPy)")
    p()
    for c in base["verify"]["checks"]:
        p(f"- {c}")
    p(f"- итог: **{'допустимо' if base['verify']['ok'] else 'ОШИБКА'}**")
    p()
    p("## 6. Узкие места сети")
    p()
    bn = [r for r in bottlenecks(g) if r["saturated"]]
    if bn:
        p("Насыщенные (достигшие пропускной способности) дуги и их "
          "теневые цены (снижение цели при добавлении единицы "
          "пропускной способности, руб./шт.):")
        p()
        p(pd.DataFrame(bn).to_markdown(index=False))
    else:
        p("Насыщенных дуг в оптимуме нет: сеть не ограничена пропускными "
          "способностями.")
    p()
    p("## 7. Сценарный анализ")
    p()
    srows = []
    for n in ["base", "deficit", "stress", "failure"]:
        r = results[n]
        c = r["ctx"]
        srows.append({"сценарий": n,
                      "статус": "optimal" if c["result"].success else "infeasible",
                      "цель, руб.": round(c["result"].fun, 2) if c["result"].success else None,
                      "проверка NumPy": "OK" if r["verify"]["ok"] else "FAIL",
                      "итераций": c["result"].nit if c["result"].success else None})
    p(pd.DataFrame(srows).to_markdown(index=False))
    p()
    f = results["failure"]
    if f["ctx"]["result"].success:
        dmg = f["ctx"]["result"].fun - res.fun
        p(f"**Отказ дуги:** закрыт маршрут "
          f"{'→'.join(f['closed_arc'])} (самый загруженный в базовом "
          f"плане). Ущерб — прирост цели на **{dmg:.2f} руб.** "
          f"({dmg / res.fun * 100:.1f}%); поток перераспределился на "
          f"обходные маршруты, дефициты магазинов перераспределились.")
    p()
    p("**Дефицитный сценарий:** запас самого большого склада сокращён на "
      "20% "
      f"({results['deficit']['changed_supply'][0]}: "
      f"{results['deficit']['changed_supply'][1]:.0f} → "
      f"{results['deficit']['changed_supply'][2]:.0f} шт.) — цель растёт "
      "за счёт переброски объёма на более дорогие маршруты и роста "
      "штрафов.")
    p()
    p("**Стрессовый сценарий:** спрос всех магазинов увеличен на 20% — "
      "дефицит углубляется, доля штрафов в цели растёт; модель остаётся "
      "совместимой, так как дефицит разрешён переменными $u_j$.")
    p()
    p("![Цель по сценариям](output/cost_by_scenario.png)")
    p()
    p("## 8. Сравнение с жадной эвристикой")
    p()
    p("Эвристика «сначала самые дешёвые маршруты»: маршруты перебираются "
      "по возрастанию тарифа, на каждом перевозится максимум из "
      "остатков запаса, спроса и пропускной способности; невыполненный "
      "спрос штрафуется.")
    p()
    p(f"- жадная эвристика: $z = {gr['objective']:.2f}$ руб.")
    p(f"- ЛП-оптимум: $z^{{*}} = {res.fun:.2f}$ руб.")
    p(f"- зазор: **{gap:.1f}%** — эвристика не учитывает, что дешёвый "
      "маршрут может «забрать» запас, нужный для более выгодной "
      "комбинации, и не планирует дефицит глобально.")
    p()
    p("## 9. Вывод")
    p()
    p(f"Минимальные затраты снабжения сети — **{res.fun:.2f} руб.** "
      f"в месяц, из которых штрафы за неудовлетворённый спрос составляют "
      f"{g['penalty'] @ u:.2f} руб. (структурный дефицит: запаса "
      f"{g['supply_used'].sum():.0f} шт. не хватает до спроса "
      f"{g['demand_used'].sum():.0f} шт.). Теневые цены насыщенных дуг "
      "показывают, где расширение пропускной способности окупается "
      "быстрее всего.")
    p()
    p("**Можно ли реализовать на практике?** Дробные потоки — "
      "усреднение за несколько периодов; для неделимых партий нужна "
      "целочисленная постановка (см. ЛР №3). Кроме того, модель не "
      "учитывает время доставки, надёжность маршрутов и сезонность "
      "спроса.")
    p()
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
