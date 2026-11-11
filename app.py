"""Robot Maze Challenge — Streamlit интерфейсі."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

from maze_ai.local_search import fitness, hill_climb
from maze_ai.maze import random_maze
from maze_ai.qlearning import QLearningAgent
from maze_ai.search import ALGORITHMS

st.set_page_config(page_title="Робот лабиринті (Robot Maze Challenge)", layout="wide")
CYBER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
.stApp {background:radial-gradient(circle at 0 0,rgba(2,150,248,.22),transparent 45%),radial-gradient(circle at 100% 100%,rgba(2,150,248,.12),transparent 45%),#0b1435;}
header[data-testid="stHeader"] {background:transparent;} .stApp::after {content:"";position:fixed;left:0;right:0;bottom:0;height:5px;background:#0296f8;z-index:999;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp button,.stApp li {font-family:Inter,sans-serif;}
[data-testid="stSidebar"] {background:#101a42;border-right:1px solid #1b2a63;box-shadow:0 12px 30px rgba(3,6,25,.45);}
.rm-title {font-family:Inter,sans-serif;font-weight:700;font-size:2.1rem;color:#e8ecff;line-height:1.15;}
.rm-title span {color:#0296f8;font-size:1.2rem;font-weight:600;}
.rm-sub {color:#9fb0e6;font-size:.95rem;margin:2px 0 18px 0;}
.stButton > button, [data-testid="stBaseButton-secondary"] {background:linear-gradient(90deg,#0762cc,#0877e0);color:#fff;border:0;border-radius:14px;padding:.6rem 1.4rem;font-weight:600;box-shadow:0 8px 20px rgba(2,150,248,.35);}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {filter:brightness(1.12);color:#fff;border:0;}
[data-baseweb="tab-list"] {gap:8px;}
button[role="tab"] {color:#9fb0e6;border-radius:14px;padding:10px 18px;font-weight:600;}
button[role="tab"][aria-selected="true"] {color:#fff;background:linear-gradient(90deg,#0762cc,#0877e0);box-shadow:0 8px 20px rgba(2,150,248,.35);}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"] {display:none;}
[data-testid="stMetric"] {background:#101a42;border:1px solid #1b2a63;border-radius:24px;padding:18px 22px;box-shadow:0 12px 30px rgba(3,6,25,.45);}
[data-testid="stMetricValue"] {color:#0296f8;font-weight:700;}
[data-testid="stMetricLabel"] {color:#9fb0e6;}
[data-testid="stAlert"] {background:#101a42;border:1px solid #1b2a63;border-radius:18px;}
"""
st.markdown("<style>" + " ".join(l.strip() for l in CYBER_CSS.splitlines() if l.strip()) + "</style>", unsafe_allow_html=True)
st.markdown(
    "<div class='rm-title'>Робот лабиринті <span>(Robot Maze Challenge)</span></div>"
    "<div class='rm-sub'>QAIRU · Жасанды интеллект негіздері (AI Fundamentals) &nbsp;—&nbsp; іздеу (A*, DFS, BFS, UCS) · төбеге өрмелеу (Hill Climbing) · Q-оқыту (Q-learning)</div>",
    unsafe_allow_html=True,
)


EXPLAIN = {
    "BFS": "Енге іздеу (BFS) қадам санын азайтады, бірақ ұяшық құнын (балшықты) ескермейді — жол оңтайлы болмауы мүмкін.",
    "DFS": "Тереңдікке іздеу (DFS) стек қолданады: жадты аз жұмсайды, бірақ жол ұзақ және оңтайлы емес болуы мүмкін.",
    "UCS": "Бірыңғай құн бойынша іздеу (UCS) жиынтық құн g(n) бойынша жүреді: құн жағынан оңтайлы, бірақ көп түйін қарастырады.",
    "A*": "Эвристикалық іздеу (A*) f(n)=g(n)+h(n) қолданады, h — Манхэттен қашықтығы (Manhattan). Оңтайлы әрі UCS-тен аз түйін қарастырады.",
}

NAMES = {
    "BFS": "Енге іздеу (BFS)",
    "DFS": "Тереңдікке іздеу (DFS)",
    "UCS": "Бірыңғай құн бойынша іздеу (UCS)",
    "A*": "Эвристикалық іздеу (A*)",
}



BG, FREE_C, WALL_C, MUD_C = "#101a42", "#0b1435", "#e8ecff", "#af7416"


def show(maze, explored=(), path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG)
    ax.imshow(np.array(maze.grid), cmap=ListedColormap([FREE_C, WALL_C, MUD_C]), vmin=0, vmax=2)
    if explored:
        ys, xs = zip(*explored)
        ax.scatter(xs, ys, c="#0296f8", s=30, alpha=0.45)
    if path:
        ys, xs = zip(*path)
        ax.plot(xs, ys, c="#0296f8", lw=3.5, solid_capstyle="round", solid_joinstyle="round")
    for p, t, col in ((maze.start, "Б", "#34d399"), (maze.goal, "М", "#fbbf24")):
        ax.scatter([p[1]], [p[0]], s=260, c=col, zorder=3)
        ax.text(p[1], p[0], t, ha="center", va="center", color="#0b1435", fontweight="bold", fontsize=11, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, maze.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.rows, 1), minor=True)
    ax.grid(which="minor", color="#172457", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_color("#1b2a63")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("<span style='color:#9fb0e6'>□</span> бос ұяшық (құны 1) &nbsp; <span style='color:#f59e0b'>■</span> балшық (құны 5) &nbsp; <span style='color:#e8ecff'>■</span> қабырға (өтпейді, құны шексіз ∞) &nbsp; "
                "<span style='color:#34d399'>●</span> бастау, Б (құны 0) &nbsp; <span style='color:#fbbf24'>●</span> мақсат, М (құны 1) &nbsp; "
                "<span style='color:#0296f8'>━</span> табылған жол", unsafe_allow_html=True)
    st.caption("Құн — ұяшыққа кіргенде төленетін баға. Жол құны = жолдағы ұяшықтар құнының қосындысы.")


with st.sidebar:
    st.header("Лабиринт")
    rows = st.slider("Жол саны", 5, 25, 12)
    cols = st.slider("Баған саны", 5, 25, 12)
    wall_p = st.slider("Қабырға тығыздығы (өтпейді)", 0.0, 0.4, 0.2, 0.05)
    mud_p = st.slider("Балшық тығыздығы (құны 5)", 0.0, 0.3, 0.1, 0.05)
    seed = int(st.number_input("Кездейсоқ сан (seed)", 0, 9999, 0))
    gen = st.button("🎲 Жаңа лабиринт")

if gen or "maze" not in st.session_state:
    st.session_state.maze = random_maze(rows, cols, wall_p, mud_p, seed)
    st.session_state.agent = None
    st.session_state.hc = None
    st.session_state.update(sr=0, sc=0, gr=rows - 1, gc=cols - 1)
maze = st.session_state.maze

with st.sidebar:
    st.subheader("Бастау және мақсат (Start / Goal)")
    maze.start = (st.number_input("Бастау: жол (Start)", 0, maze.rows - 1, key="sr"),
                  st.number_input("Бастау: баған", 0, maze.cols - 1, key="sc"))
    maze.goal = (st.number_input("Мақсат: жол (Goal)", 0, maze.rows - 1, key="gr"),
                 st.number_input("Мақсат: баған", 0, maze.cols - 1, key="gc"))
    st.header("Q-оқыту (Q-learning)")
    episodes = st.slider("Эпизодтар", 100, 3000, 800, 100)
    alpha = st.slider("α — оқу жылдамдығы (learning rate)", 0.05, 1.0, 0.5, 0.05)
    gamma = st.slider("γ — дисконт коэффициенті (discount)", 0.5, 0.99, 0.95, 0.01)
    epsilon = st.slider("ε — зерттеу ықтималдығы (exploration)", 0.0, 0.5, 0.2, 0.05)
    st.header("Төбеге өрмелеу (Hill Climbing)")
    iters = st.slider("Итерация саны", 50, 1000, 300, 50)
    restarts = st.slider("Қайта бастау саны (random restart)", 1, 5, 3)

err = maze.validate()
if err:
    st.error(err)
    st.stop()

t1, t2, t3, t4 = st.tabs(["🔍 Іздеу (Search)", "🧠 Q-оқыту (Q-learning)", "⛰️ Төбеге өрмелеу (Hill Climbing)", "📊 Салыстыру"])

with t1:
    name = st.selectbox("Алгоритм", list(ALGORITHMS), index=list(ALGORITHMS).index("A*"), format_func=lambda k: NAMES[k])
    res = ALGORITHMS[name](maze)
    st.info(EXPLAIN.get(name, ""))
    if res.path is None:
        st.warning("Жол табылмады: лабиринт жабық. Қабырғаны азайтыңыз немесе жаңа лабиринт жасаңыз.")
    k = st.slider("Қадам (қарастырылған түйіндер)", 0, len(res.expanded), len(res.expanded))
    show(maze, res.expanded[:k], res.path if k == len(res.expanded) else None)
    c = st.columns(3)
    c[0].metric("Жол құны", res.cost if res.path else "—")
    c[1].metric("Қарастырылған түйін", len(res.expanded))
    c[2].metric("Уақыт, мс", f"{res.time_ms:.3f}")

with t2:
    st.info("Агент лабиринтті алдын ала білмейді: сыйақы арқылы Q(s,a) кестесін жаңартып үйренеді. "
            "Қадам сыйақысы = −ұяшық құны, мақсат +100, қабырғаға соғылу −5.")
    if st.button("▶ Оқытуды бастау"):
        agent = QLearningAgent(maze, alpha, gamma, epsilon, seed)
        t0 = time.perf_counter()
        rewards, steps = agent.train(episodes)
        st.session_state.agent = (agent, rewards, steps, time.perf_counter() - t0, (maze.start, maze.goal))
    ag = st.session_state.agent
    if ag is None:
        st.write("Параметрлерді таңдап, оқытуды бастаңыз.")
    elif ag[4] != (maze.start, maze.goal):
        st.warning("Бастау не мақсат нүктесі өзгерді (Start / Goal) — агентті қайта оқытыңыз.")
    else:
        path = ag[0].greedy_path()
        if path is None:
            st.warning("Агент жолды әлі үйренбеді: эпизод санын көбейтіңіз.")
        show(maze, path=path)
        st.caption("Эпизод сыйақысы (20 эпизодтық орташа): өсу = агент үйреніп жатыр")
        st.line_chart(pd.Series(ag[1]).rolling(20, min_periods=1).mean())

with t3:
    st.info("Мақсат: A* үшін ең қиын лабиринтті табу. Бағалау функциясы (fitness) = A* қарастырған түйін саны (−1 = жол жоқ). "
            "Бір ұяшықты өзгертеді, тек жақсарса қабылдайды; қайта бастау (random restart) жергілікті максимумнан шығуға көмектеседі.")
    if st.button("⛰️ Төбеге өрмелеуді іске қосу"):
        before = fitness(maze)
        best, hist = hill_climb(maze, iters, restarts, seed)
        st.session_state.maze, st.session_state.agent = best, None
        st.session_state.hc = (before, hist)
        st.rerun()
    if st.session_state.hc:
        before, hist = st.session_state.hc
        st.metric("Бағалау (fitness): бұрын → кейін", f"{before} → {hist[-1]}")
        st.line_chart(pd.Series(hist))
        st.caption("Жаңа лабиринт барлық қойындыларда қолданылады.")

with t4:
    table = []
    for n, f in ALGORITHMS.items():
        r = f(maze)
        table.append({"Алгоритм": NAMES[n], "Жол құны": r.cost, "Қадам": len(r.path) - 1 if r.path else None,
                      "Қарастырылған түйін": len(r.expanded), "Уақыт, мс": round(r.time_ms, 3)})
    ag = st.session_state.agent
    if ag and ag[4] == (maze.start, maze.goal):
        p = ag[0].greedy_path()
        cost = sum(maze.cost(x) for x in p[1:]) if p else None
        table.append({"Алгоритм": f"Q-оқыту (Q-learning), {len(ag[1])} эпизод", "Жол құны": cost,
                      "Қадам": len(p) - 1 if p else None, "Қарастырылған түйін": sum(ag[2]),
                      "Уақыт, мс": round(ag[3] * 1000, 1)})
    st.dataframe(pd.DataFrame(table), use_container_width=True)
    st.caption("Q-оқыту үшін «түйін» = оқыту кезіндегі барлық қадам саны. Эвристикалық іздеу (A*) мен UCS оңтайлы жолды бірден табады, "
               "Q-оқыту көп эпизодтан кейін ғана оңтайлы жолға жақындайды.")
