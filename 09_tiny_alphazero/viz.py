"""Demos: a game GIF (with the net's search distribution under the board) and a
'visible search tree' (self-contained HTML + PNG) for an interesting position.

  python viz.py gif  --a net:checkpoints/main/gen_080:200 --b ab8 --out media/final_vs_ab8.gif
  python viz.py tree --ckpt checkpoints/main/gen_080 --sims 800 --out media/search_tree
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import arena
import c4
from c4 import W, H
from mcts import BatchMCTS

HERE = os.path.dirname(os.path.abspath(__file__))
RED, YEL, BLUE, BG = (220, 50, 47), (245, 200, 40), (30, 80, 170), (250, 248, 240)


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_frame(grid, title, sub, bars=None, bar_label="", last=None, win_cells=()):
    cell = 64
    top, bottom = 96, 120
    img = Image.new("RGB", (W * cell + 40, top + H * cell + bottom), BG)
    d = ImageDraw.Draw(img)
    f1, f2, f3 = _font(17), _font(14), _font(12)
    if isinstance(title, tuple):          # (red player, yellow player)
        for k, (nm, col) in enumerate(zip(title, (RED, YEL))):
            y = 8 + 24 * k
            d.ellipse([20, y + 3, 34, y + 17], fill=col)
            d.text((42, y), nm, fill=(30, 30, 30), font=f1)
    else:
        d.text((20, 10), title, fill=(30, 30, 30), font=f1)
    d.text((20, 62), sub, fill=(80, 80, 80), font=f2)
    x0, y0 = 20, top
    d.rounded_rectangle([x0, y0, x0 + W * cell, y0 + H * cell], 10, fill=BLUE)
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            col = BG if v == 0 else (RED if v == 1 else YEL)
            cx, cy = x0 + c * cell + cell // 2, y0 + r * cell + cell // 2
            rad = cell // 2 - 6
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)
            if (r, c) in win_cells:
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(20, 20, 20), width=4)
            if last == (r, c):
                d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 255))
    if bars is not None:
        by = y0 + H * cell + 12
        maxh = 60
        bmax = max(float(np.max(bars)), 1e-9)
        for c in range(W):
            h = int(maxh * bars[c] / bmax)
            bx = x0 + c * cell + 14
            d.rectangle([bx, by + maxh - h, bx + cell - 28, by + maxh], fill=(90, 90, 90))
            d.text((bx, by + maxh + 4), f"{bars[c]:.0%}" if bars[c] > 0 else "", fill=(60, 60, 60), font=f3)
        d.text((x0, by + maxh + 22), bar_label, fill=(80, 80, 80), font=f3)
    return img


def play_recorded(a, b, opening, seed=0):
    """One game a (first, red) vs b (second, yellow). Returns frames info."""
    c4.seed_numba(seed)
    pos, mask = c4.from_moves(opening)
    moves = list(opening)
    steps = []
    agents = [a, b]
    ply = len(opening)
    while True:
        ag = agents[ply % 2]
        mv = int(ag.moves(np.array([pos]), np.array([mask]))[0])
        info = None
        if isinstance(ag, arena.MCTSAgent) and ag.last is not None and ag.sims > 0:
            v, q, p = ag.last
            info = dict(visits=(v[0] / v[0].sum()).tolist(), prior=p[0].tolist(),
                        q=q[0].tolist())
        won = c4.winning_move(pos, mask, mv)
        pos, mask = c4.play(pos, mask, mv)
        moves.append(mv)
        steps.append(dict(move=mv, mover=ply % 2, info=info, pos=int(pos), mask=int(mask)))
        ply += 1
        if won:
            return moves, steps, (1 if (ply - 1) % 2 == 0 else -1)
        if c4.is_full(mask):
            return moves, steps, 0


def win_cells(pos, mask):
    g = c4.to_grid(pos, mask)
    for r in range(H):
        for c in range(W):
            p = g[r, c]
            if not p:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + k * dr, c + k * dc) for k in range(4)]
                if all(0 <= rr < H and 0 <= cc < W and g[rr, cc] == p for rr, cc in cells):
                    return set(cells)
    return set()


def make_gif(a_spec, b_spec, out, opening=(), seed=0, names=None):
    a = arena.make_agent(a_spec, max_games=1, seed=seed)
    b = arena.make_agent(b_spec, max_games=1, seed=seed + 1)
    na, nb = names or (a_spec, b_spec)
    moves, steps, res = play_recorded(a, b, list(opening), seed)
    title = (na, nb)
    frames = []
    p0, m0 = c4.from_moves(list(opening))
    frames.append(draw_frame(c4.to_grid(p0, m0), title, "opening: " + (",".join(map(str, opening)) or "empty board")))
    for i, st in enumerate(steps):
        g = c4.to_grid(st["pos"], st["mask"])
        col = st["move"]
        row = int(np.nonzero(g[:, col])[0][0])
        who = "red" if st["mover"] == 0 else "yellow"
        sub = f"move {len(opening) + i + 1}: {who} plays column {col}"
        bars, bl = None, ""
        if st["info"]:
            bars = np.array(st["info"]["visits"])
            qv = st["info"]["q"][col]
            bl = f"{who}'s search: share of visits per column (Q of chosen move {qv:+.2f})"
        last = i == len(steps) - 1
        wc = win_cells(st["pos"], st["mask"]) if last and res != 0 else ()
        if last:
            sub += "  —  " + ("draw" if res == 0 else f"{'red' if res == 1 else 'yellow'} wins")
        frames.append(draw_frame(g, title, sub, bars, bl, (row, col), wc))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    durs = [900] + [700] * (len(frames) - 2) + [3000]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0)
    with open(out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(dict(a=a_spec, b=b_spec, opening=list(opening), moves=moves, result=res), f)
    print(f"wrote {out}: {len(moves)} plies, result {res} (+1 = red/first wins)")
    return res


# ---------------------------------------------------------------- search tree
def pick_interesting(evaluate, sims, n_games=24, seed=0):
    """Play a few net-vs-ab6 games and return the position where search disagrees most
    with the raw policy (total variation between root visit share and prior)."""
    rng = np.random.default_rng(seed)
    best = None
    m = BatchMCTS(1, sims)
    ab = arena.make_agent("ab6")
    for gi in range(n_games):
        pos, mask = c4.from_moves([int(rng.integers(W)), int(rng.integers(W))])
        moves = []
        net_turn = gi % 2
        k = 0
        while not c4.status(pos, mask):
            if k % 2 == net_turn:
                v, q, p = m.search(np.array([pos]), np.array([mask]), evaluate)
                share = v[0] / v[0].sum()
                tv = 0.5 * np.abs(share - p[0]).sum()
                if c4.n_stones(mask) >= 6 and (best is None or tv > best[0]):
                    best = (tv, int(pos), int(mask))
                a = int(np.argmax(v[0]))
            else:
                a = int(ab.moves(np.array([pos]), np.array([mask]))[0])
            pos, mask = c4.play(pos, mask, a)
            k += 1
    return best


def tree_to_svg_html(tree, title, note):
    data = json.dumps(tree)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visible search tree</title>
<style>
:root{{--bg:#faf8f0;--fg:#222;--muted:#666;--edge:#bbb;--red:#dc322f;--yel:#f5c828;--blue:#1e50aa;--card:#fff}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16181d;--fg:#e8e6e0;--muted:#9a9a9a;--edge:#555;--card:#22252c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,system-ui,sans-serif}}
header{{padding:16px}} h1{{font-size:20px;margin:0 0 4px}} p{{margin:4px 0;color:var(--muted);max-width:900px}}
#wrap{{display:flex;gap:16px;flex-wrap:wrap;padding:0 16px 16px}}
#tree{{flex:1 1 640px;overflow:auto;background:var(--card);border-radius:10px}}
#side{{flex:0 0 260px;background:var(--card);border-radius:10px;padding:12px}}
.node circle{{cursor:pointer;stroke:var(--fg);stroke-width:1}} .node text{{font-size:10px;fill:var(--fg);pointer-events:none}}
.edge{{stroke:var(--edge);fill:none}}
table{{border-collapse:collapse;font-size:12px}} td{{padding:2px 6px}}
</style></head><body>
<header><h1>{title}</h1><p>{note}</p>
<p>Circle area ∝ visit count N. Colour = Q from the view of the player who made that move (green good, red bad). Label: column, N, Q, prior P. Click a node to see the board there; click again to collapse/expand its children. Only the most-visited children are drawn (all 7 at the root, then the top 4, 3, 2 by depth).</p></header>
<div id="wrap"><div id="tree"><svg id="svg"></svg></div><div id="side"><div id="info"></div><svg id="board" width="236" height="206"></svg></div></div>
<script>
const T={data};
const W=7,H=6,H1=7;
function grid(pos,mask){{const n=popc(mask);const firstToMove=n%2==0;const me=BigInt(pos),opp=BigInt(pos)^BigInt(mask);
 const first=firstToMove?me:opp, second=firstToMove?opp:me; const g=[];
 for(let r=0;r<H;r++){{g.push([]);for(let c=0;c<W;c++){{const b=1n<<BigInt(c*H1+(H-1-r));g[r].push((first&b)?1:((second&b)?2:0));}}}} return g;}}
function popc(m){{let x=BigInt(m),n=0;while(x){{x&=x-1n;n++;}}return n;}}
function drawBoard(node){{const s=document.getElementById('board');const g=grid(node.pos,node.mask);let h='<rect x="0" y="0" width="236" height="206" rx="8" fill="#1e50aa"/>';
 for(let r=0;r<H;r++)for(let c=0;c<W;c++){{const v=g[r][c];h+=`<circle cx="${{18+c*33.3}}" cy="${{18+r*33.6}}" r="13" fill="${{v==1?'#dc322f':v==2?'#f5c828':'#faf8f0'}}"/>`;}}
 s.innerHTML=h;
 const path=[];let p=node;while(p){{if(p.move!==null)path.unshift(p.move);p=p._parent;}}
 document.getElementById('info').innerHTML=`<b>path from root:</b> ${{path.length?path.join(' → '):'(root)'}}<br><table><tr><td>N</td><td>${{node.N}}</td></tr><tr><td>Q</td><td>${{node.Q===null?'–':node.Q.toFixed(3)}}</td></tr><tr><td>P</td><td>${{node.P.toFixed(3)}}</td></tr><tr><td>terminal</td><td>${{node.term==1?'win for mover':node.term==2?'draw':'no'}}</td></tr></table>`;}}
function prep(n,parent){{n._parent=parent;n._open=true;(n.children||[]).forEach(c=>prep(c,n));}}
prep(T,null);
function layout(){{let x=0;const dy=95;function rec(n,d){{n._d=d;const kids=n._open?n.children:[];if(!kids.length){{n._x=x++;}}else{{kids.forEach(k=>rec(k,d+1));n._x=(kids[0]._x+kids[kids.length-1]._x)/2;}}}}rec(T,0);return x;}}
function color(q){{if(q===null)return '#999';const t=(q+1)/2;const r=Math.round(220*(1-t)+40*t),g=Math.round(60*(1-t)+170*t);return `rgb(${{r}},${{g}},70)`;}}
function render(){{const nleaf=layout();const dx=Math.max(46,Math.min(90,1100/nleaf));const svg=document.getElementById('svg');let e='',nodes='';let maxd=0;
 const rootN=T.N;
 function rec(n){{maxd=Math.max(maxd,n._d);const x=30+n._x*dx,y=40+n._d*95;const kids=n._open?n.children:[];
  kids.forEach(k=>{{const kx=30+k._x*dx,ky=40+k._d*95;e+=`<path class="edge" d="M${{x}},${{y}} C${{x}},${{y+45}} ${{kx}},${{ky-45}} ${{kx}},${{ky}}" stroke-width="${{1+6*k.N/rootN}}"/>`;rec(k);}});
  const r=4+18*Math.sqrt(n.N/rootN);
  nodes+=`<g class="node" transform="translate(${{x}},${{y}})" data-id="${{n._id}}"><circle r="${{r}}" fill="${{color(n.Q)}}"/>`+
   (n.move===null?`<text y="-${{r+4}}" text-anchor="middle">root N=${{n.N}}</text>`:`<text y="${{r+11}}" text-anchor="middle">c${{n.move}} N${{n.N}}</text><text y="${{r+22}}" text-anchor="middle">Q${{n.Q===null?'–':n.Q.toFixed(2)}} P${{n.P.toFixed(2)}}</text>`)+`</g>`;}}
 rec(T);svg.setAttribute('width',60+nleaf*dx);svg.setAttribute('height',80+(maxd+1)*95);svg.innerHTML=e+nodes;
 svg.querySelectorAll('.node').forEach(g=>g.addEventListener('click',()=>{{const n=byId[g.dataset.id];drawBoard(n);if(n.children&&n.children.length){{n._open=!n._open;render();}}}}));}}
const byId={{}};let nid=0;(function idx(n){{n._id=nid;byId[nid++]=n;(n.children||[]).forEach(idx);}})(T);
render();drawBoard(T);{{const t=document.getElementById("tree");t.scrollLeft=Math.max(0,30+T._x*Math.max(46,Math.min(90,1100/layout()))-t.clientWidth/2);}}
</script></body></html>"""


def tree_png(tree, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leaves = []

    def lay(n, d):
        n["_d"] = d
        if not n["children"]:
            n["_x"] = len(leaves)
            leaves.append(n)
        else:
            for k in n["children"]:
                lay(k, d + 1)
            n["_x"] = (n["children"][0]["_x"] + n["children"][-1]["_x"]) / 2
    lay(tree, 0)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(leaves)), 6))
    root_n = tree["N"]
    cmap = plt.get_cmap("RdYlGn")

    def draw(n):
        for k in n["children"]:
            ax.plot([n["_x"], k["_x"]], [-n["_d"], -k["_d"]], color="#bbb", lw=0.5 + 5 * k["N"] / root_n, zorder=1)
            draw(k)
        q = n["Q"] if n["Q"] is not None else 0
        ax.scatter([n["_x"]], [-n["_d"]], s=20 + 1500 * n["N"] / root_n, color=cmap((q + 1) / 2),
                   edgecolor="k", lw=0.5, zorder=2)
        if n["move"] is None:
            ax.text(n["_x"], -n["_d"] + 0.18, f"root N={int(n['N'])}", ha="center", fontsize=8)
        else:
            ax.text(n["_x"], -n["_d"] - 0.22, f"c{n['move']}\nN{int(n['N'])}\nQ{q:+.2f}\nP{n['P']:.2f}",
                    ha="center", va="top", fontsize=6)
    draw(tree)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    ax.set_ylim(-(max(l["_d"] for l in leaves) + 0.9), 0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def prune(tree, widths):
    """Keep the top-k children (by visits) at each depth so the picture stays readable."""
    def rec(n, d):
        kids = sorted(n["children"], key=lambda k: -k["N"])
        n["children"] = kids[:widths[d]] if d < len(widths) else []
        n["children"].sort(key=lambda k: k["move"])
        for k in n["children"]:
            rec(k, d + 1)
    rec(tree, 0)
    return tree


def make_tree(ckpt, sims, out_prefix, seed=0):
    from net import load_model, make_evaluator
    model, meta = load_model(ckpt)
    ev = make_evaluator(model)
    c4.seed_numba(seed)
    tv, pos, mask = pick_interesting(ev, 200, n_games=8, seed=seed)
    m = BatchMCTS(1, sims)
    v, q, p = m.search(np.array([pos], np.int64), np.array([mask], np.int64), ev)
    tree = m.export_tree(0, max_depth=4, min_visits=1)
    tree_html_t = prune(json.loads(json.dumps(tree)), [7, 4, 3, 2])
    to_move = "red (first player)" if c4.n_stones(np.int64(mask)) % 2 == 0 else "yellow (second player)"
    share = v[0] / v[0].sum()
    note = (f"Checkpoint {ckpt} (generation {meta.get('gen')}), {sims} simulations, {to_move} to move. "
            f"This position was picked automatically as the one where search most changed the raw policy "
            f"(total variation {tv:.2f} between prior and visit share in a 200-sim search, scanned over 8 games vs alpha-beta depth 6). Prior: "
            + ", ".join(f"c{c} {p[0][c]:.2f}" for c in range(W)) + ". Visit share: "
            + ", ".join(f"c{c} {share[c]:.2f}" for c in range(W)) + ".")
    with open(out_prefix + ".html", "w") as f:
        f.write(tree_to_svg_html(tree_html_t, "Visible search tree — Tiny AlphaZero", note))
    tree_png(prune(json.loads(json.dumps(tree)), [7, 3, 2]), out_prefix + ".png",
             f"MCTS tree ({sims} sims), gen {meta.get('gen')}, {to_move} to move — top children by visits")
    # board image of the root position
    img = draw_frame(c4.to_grid(pos, mask), "Search-tree root position", f"{to_move} to move",
                     share, "share of MCTS visits per column")
    img.save(out_prefix + "_board.png")
    print(f"wrote {out_prefix}.html/.png; tv={tv:.2f}")
    print(c4.render_ascii(pos, mask))
    return pos, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gif", "tree"])
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--names", default=None, help="comma-separated display names")
    ap.add_argument("--opening", default="")
    ap.add_argument("--ckpt")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "gif":
        op = [int(x) for x in a.opening.split(",") if x]
        make_gif(a.a, a.b, a.out, op, a.seed, a.names.split(",") if a.names else None)
    else:
        make_tree(a.ckpt, a.sims, a.out, a.seed)


if __name__ == "__main__":
    main()
