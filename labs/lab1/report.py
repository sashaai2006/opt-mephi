from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"


def solution_df(sheets, items, patterns, ctx, ver):
    res = ctx["result"]
    if not res.success:
        return pd.DataFrame()
    df = patterns.copy()
    df["sheets_used"] = res.x
    df = df[df["sheets_used"] > 1e-9].merge(
        sheets[["sheet_id", "cost", "name"]], on="sheet_id")
    df["cost_total"] = df["sheets_used"] * df["cost"]
    kcols = [f"k_{iid}" for iid in items["item_id"]]
    df["pattern_desc"] = df.apply(
        lambda r: "+".join(f"{int(r[c])}x{c.split('_', 1)[1]}" for c in kcols
                           if r[c] > 0), axis=1)
    return df[["pattern_id", "name", "pattern_desc", "sheets_used", "cost_total"]]


def plots(results):
    OUT.mkdir(exist_ok=True)
    names = list(results)
    costs = [results[n]["ctx"]["result"].fun if results[n]["ctx"]["result"].success
             else np.nan for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, costs, color=["tab:blue", "tab:orange", "tab:red"])
    ax.set_ylabel("стоимость, руб.")
    ax.set_title("Целевая функция по сценариям")
    for k, v in enumerate(costs):
        if not np.isnan(v):
            ax.text(k, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "cost_by_scenario.png", dpi=150)
    plt.close(fig)

    base = results["base"]["ctx"]
    labels = base["sheet_ids"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, n in enumerate(names):
        ctx = results[n]["ctx"]
        if ctx["result"].success:
            used = ctx["S"] @ ctx["result"].x
            ax.bar(x + (k - 1) * width, used, width, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("листов, шт.")
    ax.set_title("Расход листов по типам и сценариям")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sheets_by_scenario.png", dpi=150)
    plt.close(fig)


def report(sheets, items, patterns, results):
    OUT.mkdir(exist_ok=True)
    L = []

    def p(s=""):
        L.append(s)
    base = results["base"]
    bctx, bver = base["ctx"], base["verify"]
    res = bctx["result"]

    p("# ЛР №1. План раскроя листового материала (вариант 10)")
    p()
    p(f"seed генерации данных: **10217** "
      f"(N = 17, Q = 2 — группа 527, Y = 26). Решатель: HiGHS "
      f"(scipy.optimize.linprog, method='highs').")
    p()
    p("## 1. Постановка и вариант")
    p()
    p("Мебельное производство заказывает раскрой листового материала: "
      "4 типа листов (размеры, цена за лист, ограниченный запас на складе) "
      "и 5 позиций деталей (размеры, спрос). Требуется минимизировать "
      "стоимость израсходованных листов, покрыв спрос.")
    p()
    p("Исходные данные (data/sheets.csv, data/items.csv, data/patterns.csv):")
    p()
    sh = sheets.copy()
    p(sh.to_markdown(index=False))
    p()
    it = items.copy()
    p(it.to_markdown(index=False))
    p()
    p("## 2. Формальная модель (усреднённая ЛП-постановка)")
    p()
    p("- Множества: листы $t \\in T$ (|T| = 4), детали $i \\in I$ (|I| = 5), "
      "паттерны $p \\in P_t$ — допустимые расклады деталей в ряд вдоль длины "
      "листа типа $t$; $a_{ip}$ — число деталей $i$ в паттерне $p$.")
    p("- Переменные: $x_p \\ge 0$ — число листов, раскроенных по паттерну $p$ "
      "(непрерывные, шт.; допущение усреднённой постановки).")
    p("- Параметры: $c_t$ — цена листа типа $t$ (руб.), $d_i$ — спрос (шт.), "
      "$s_t$ — запас листов типа $t$ (шт.), $c_p = c_{t(p)}$.")
    p("- Цель: $\\min z = \\sum_p c_p x_p$ (руб.).")
    p("- Ограничения: покрытие спроса $\\sum_p a_{ip} x_p \\ge d_i$ "
      "(шт. деталей), запас $\\sum_{p \\in P_t} x_p \\le s_t$ (шт. листов), "
      "$x \\ge 0$.")
    p()
    p("**Допущения:** детали раскладываются в один ряд вдоль длины листа "
      "(полосовая раскладка), обрезь не учитывается; листы делимы "
      "(усреднённая постановка); спрос и запасы детерминированы; толщина и "
      "направление волокон не учитываются.")
    p()
    p("## 3. Реализация и статус решения")
    p()
    p(f"Модель построена в SciPy, решатель HiGHS (двойственный симплекс, "
      f"presolve включён). Статус: **{res.status} — оптимально**; "
      f"итераций: {res.nit}; время: {bctx['elapsed']:.3f} с; "
      f"паттернов в модели: {bctx['A'].shape[1]}. "
      f"Лог солвера: output/solver_log.txt.")
    p()
    p(f"**Оптимум: $z^{{*}} = {res.fun:.2f}$ руб.**")
    p()
    p("## 4. Найденное решение (предметные термины)")
    p()
    sol = solution_df(sheets, items, patterns, bctx, bver)
    sol_round = sol.copy()
    sol_round["sheets_used"] = sol_round["sheets_used"].round(3)
    sol_round["cost_total"] = sol_round["cost_total"].round(2)
    p(sol_round.to_markdown(index=False))
    p()
    used = bctx["S"] @ res.x
    usage = pd.DataFrame({"тип": bctx["sheet_ids"], "использовано": used.round(3),
                          "запас": bctx["supply"],
                          "остаток": bver["slack_supply"].round(3)})
    p(usage.to_markdown(index=False))
    p()
    p("## 5. Независимая проверка (NumPy, без объекта решения)")
    p()
    for name, ok in bver["checks"]:
        p(f"- {name}: {'OK' if ok else 'FAIL'}")
    p(f"- пересчёт цели: {bver['objective_recomputed']:.2f} руб., "
      f"совпадает с солвером: {'да' if bver['objective_matches_solver'] else 'нет'}")
    p(f"- итог проверки: **{'допустимо' if bver['ok'] else 'ошибка'}**")
    p()
    p("## 6. Теневые цены и сценарный анализ")
    p()
    mb = base["marginals"]
    dem = pd.DataFrame({"деталь": items["item_id"],
                        "теневая цена спроса, руб./шт.": np.round(mb["demand"], 2)})
    sup = pd.DataFrame({"тип листа": bctx["sheet_ids"],
                        "теневая цена запаса, руб./шт.": np.round(-mb["supply"], 2)})
    p("Теневые цены активных ограничений (базовый сценарий):")
    p()
    p(dem.to_markdown(index=False))
    p()
    p(sup.to_markdown(index=False))
    p()
    p("Сценарии (таблица также в output/scenarios.csv):")
    p()
    rows = []
    for n in ["base", "deficit", "stress"]:
        r = results[n]
        ctx = r["ctx"]
        ok = ctx["result"].success
        rows.append({
            "сценарий": n,
            "статус": "optimal" if ok else "infeasible",
            "цель, руб.": round(ctx["result"].fun, 2) if ok else None,
            "проверка NumPy": "OK" if r["verify"]["ok"] else "FAIL",
        })
    p(pd.DataFrame(rows).to_markdown(index=False))
    p()
    if "changed_supply" in results["deficit"]:
        sid, s0, s1 = results["deficit"]["changed_supply"]
        p(f"**Дефицитный сценарий:** запас листов {sid} сокращён на 20% "
          f"({s0:.0f} → {s1:.0f} шт.).")
    if "changed_demand" in results["stress"]:
        chg = "; ".join(f"{iid}: {d0:.0f} → {d1:.0f} шт."
                        for iid, d0, d1 in results["stress"]["changed_demand"])
        p(f"**Стрессовый сценарий:** то же дефицитное ограничение запаса плюс "
          f"рост спроса на 20% по двум позициям ({chg}).")
    p()
    p("Сверка прогноза теневой цены с фактом (изменение цели при релаксации "
      "ограничения на величину $\\Delta$):")
    p()
    from .scenarios import shadow_price_check
    sp = shadow_price_check(results)
    if sp:
        sp_rows = [{"ограничение": n, "$\\Delta$": dv,
                    "прогноз $\\Delta z$": round(pr, 2),
                    "факт $\\Delta z$": round(ac, 2)} for n, dv, pr, ac in sp]
        p(pd.DataFrame(sp_rows).to_markdown(index=False))
    p()
    p("![Целевая функция по сценариям](output/cost_by_scenario.png)")
    p()
    p("![Расход листов по сценариям](output/sheets_by_scenario.png)")
    p()
    p("## 7. Нулевые переменные и их смысл")
    p()
    n_pos = int(np.sum(res.x > 1e-9))
    p(f"Из {len(res.x)} паттернов в оптимуме используется {n_pos}; остальные "
      "равны нулю. Это типично для ЛП: оптимум достигается в вершине, где "
      "базис задаёт лишь несколько паттернов — конкурентоспособны те, что "
      "дают низкую себестоимость детали и хорошее заполнение листа; "
      "прочие паттерны доминируются и не запускаются. Предметно: освоение "
      "дополнительных схем раскроя не окупается.")
    p()
    p("## 8. Вывод")
    p()
    p(f"Минимальная стоимость раскроя — **{res.fun:.2f} руб.** План: "
      "используются паттерны из таблицы раздела 4; запасы листов "
      "расходуются частично (см. остатки). Теневые цены показывают, что "
      "выгоднее всего наращивать спрос на детали с высокой ценой "
      "в пересчёте и запасы дефицитных типов листов.")
    p()
    p("**Можно ли реализовать на практике?** Усреднённая постановка "
      "допускает дробные листы, что на практике означает усреднение за "
      "несколько производственных циклов; для единичного заказа решение "
      "потребовало бы целочисленного округления (МИЛП) и учёта обрези, "
      "переналадок и направления волокон — это ограничения модели, которым "
      "посвящена ЛР №3.")

    (OUT / "report.md").write_text("\n".join(L) + "\n")
    return sol
