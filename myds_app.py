# =========================
# Professor CoderBuddy AI — DSA No-Code Coach (Merged, Clean, Complete)
# =========================
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque, defaultdict
import heapq
from textwrap import dedent

# -------------------------------------------------------
# App Meta & simple CSS accents
# -------------------------------------------------------
st.set_page_config(
    page_title="Professor CoderBuddy AI • DSA No-Code Concept Coach",
    page_icon="🎓",
    layout="wide",
)

W3CSS = """
<style>
h1, h2, h3 { font-weight: 700; color: #0f172a; }
.badge { display:inline-block; padding:4px 10px; border:1px solid #cbd5e1;
         border-radius:999px; background:#f8fafc; color:#0f172a; margin-right:6px;
         margin-bottom:6px; font-size:0.85rem; }
.card { border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; background:#fff; margin:8px 0; }
.card .title { font-weight:700; margin-bottom:6px; }
.card.note  { border-left:6px solid #06b6d4; }
.card.tip   { border-left:6px solid #22c55e; }
.card.warn  { border-left:6px solid #f59e0b; }
.card.exam  { border-left:6px solid #8b5cf6; }
.card.try   { border-left:6px solid #0ea5e9; }
</style>
"""
st.markdown(W3CSS, unsafe_allow_html=True)

# -------------------------------------------------------
# Small UI helpers
# -------------------------------------------------------
def chip(text: str):
    st.markdown(
        f"""<span style="display:inline-block;padding:4px 10px;border:1px solid #cbd5e1;border-radius:999px;background:#f8fafc;color:#0f172a;margin-right:6px;margin-bottom:6px;font-size:0.85rem;">{text}</span>""",
        unsafe_allow_html=True
    )

def titled_box(title: str, body: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin:8px 0;background:#ffffff">
            <div style="font-weight:700;margin-bottom:6px">{title}</div>
            <div style="color:#334155">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def badge(text: str):
    st.markdown(f'<span class="badge">{text}</span>', unsafe_allow_html=True)

def card(kind: str, title: str, body: str):
    st.markdown(
        f'<div class="card {kind}"><div class="title">{title}</div><div>{body}</div></div>',
        unsafe_allow_html=True
    )

def start_canvas(figsize=(8, 2.6), hide_axes=True):
    fig, ax = plt.subplots(figsize=figsize)
    if hide_axes:
        ax.axis("off")
    return fig, ax

# -------------------------------------------------------
# Reusable Visuals (helpers)
# -------------------------------------------------------
def draw_array(ax, arr: List[Any], highlight: Optional[List[int]] = None, title: str = "", cell_w=1.3):
    for i, v in enumerate(arr):
        x = i * cell_w
        ax.add_patch(Rectangle((x, 0), cell_w, 1, fill=False, linewidth=2))
        ax.text(x + cell_w/2, 0.5, str(v), ha="center", va="center", fontsize=12)
        ax.text(x + cell_w/2, -0.3, f"{i}", ha="center", va="center", fontsize=9, color="#64748b")
        if highlight and i in highlight:
            ax.add_patch(Rectangle((x, 0), cell_w, 1, fill=False, linewidth=3))
    ax.set_xlim(-0.4, max(6, len(arr)*cell_w) + 0.4)
    ax.set_ylim(-0.6, 1.3)
    if title:
        ax.set_title(title, fontsize=13, pad=8)

def draw_two_pointers(ax, arr: List[int], i: int, j: int, title: str = ""):
    draw_array(ax, arr, highlight=[i, j])
    x_i, x_j = i*1.3 + 0.65, j*1.3 + 0.65
    ax.text(x_i, 1.2, "← i", ha="center", fontsize=11)
    ax.text(x_j, 1.2, "j →", ha="center", fontsize=11)
    if title: ax.set_title(title, fontsize=13, pad=8)

def draw_window(ax, arr: List[int], l: int, r: int, title: str = ""):
    draw_array(ax, arr)
    x0 = l*1.3; w = (r-l+1)*1.3
    ax.add_patch(Rectangle((x0, 0), w, 1, fill=False, linewidth=3))
    ax.text(x0 + w/2, 1.2, f"window [{l},{r}]", ha="center", fontsize=11)
    if title: ax.set_title(title, fontsize=13, pad=8)

def draw_prefix(ax, arr: List[int], title: str = ""):
    draw_array(ax, arr, title="Array")
    pref = np.cumsum(arr).tolist()
    y2 = -1.8; cell_w = 1.3
    for i, v in enumerate(pref):
        x = i * cell_w
        ax.add_patch(Rectangle((x, y2), cell_w, 1, fill=False, linewidth=2))
        ax.text(x + cell_w/2, y2 + 0.5, str(v), ha="center", va="center", fontsize=12)
        ax.text(x + cell_w/2, y2 - 0.3, f"{i}", ha="center", va="center", fontsize=9, color="#64748b")
    ax.text(-0.5, -1.3, "Prefix:", fontsize=11)
    ax.set_ylim(-3.0, 1.4)
    if title: ax.set_title(title, fontsize=13, pad=8)

def draw_chars(ax, s: str, highlight: Optional[int] = None, title: str = ""):
    draw_array(ax, list(s), highlight=[highlight] if highlight is not None else None, title=title)

@dataclass
class TNode:
    data: int
    left: Optional["TNode"]=None
    right: Optional["TNode"]=None

def example_tree() -> TNode:
    r = TNode(7)
    r.left = TNode(3, TNode(1), TNode(5))
    r.right = TNode(9, None, TNode(10))
    return r

def tree_height(r: Optional[TNode]) -> int:
    if not r: return 0
    return 1 + max(tree_height(r.left), tree_height(r.right))

def draw_tree(ax, r: Optional[TNode], title=""):
    xs = {}; i = [0]
    def setx(n):
        if not n: return
        setx(n.left); xs[id(n)] = i[0]; i[0]+=1; setx(n.right)
    setx(r)
    def draw(n, d=0):
        if not n: return
        x = xs[id(n)]; y = -d
        ax.add_patch(Circle((x, y), 0.25, fill=False, linewidth=2))
        ax.text(x, y, str(n.data), ha="center", va="center", fontsize=11)
        if n.left:
            xl, yl = xs[id(n.left)], -(d+1)
            ax.plot([x, xl], [y-0.25, yl+0.25], linewidth=1.8)
            draw(n.left, d+1)
        if n.right:
            xr, yr = xs[id(n.right)], -(d+1)
            ax.plot([x, xr], [y-0.25, yr+0.25], linewidth=1.8)
            draw(n.right, d+1)
    H = tree_height(r)
    ax.set_xlim(-1, max(4, len(xs)+1)); ax.set_ylim(-(H+1), 1); ax.axis("off")
    draw(r)
    if title: ax.set_title(title, fontsize=13, pad=8)

def draw_heap(ax, arr: List[int], title="Min-Heap: parent ≤ children"):
    if not arr:
        ax.text(0,0,"(empty)", fontsize=12); return
    n = len(arr); positions = {}; level = 0; idx = 0; xgap = 1.6
    while idx < n:
        count = min(2**level, n-idx)
        xs = np.linspace(0, (count-1)*xgap, count)
        for k in range(count): positions[idx] = (xs[k], -level); idx += 1
        level += 1
    for i, v in enumerate(arr):
        x,y = positions[i]
        ax.add_patch(Circle((x,y), 0.22, fill=False, linewidth=2))
        ax.text(x, y, str(v), ha="center", va="center", fontsize=11)
        if i>0:
            p = (i-1)//2; xp, yp = positions[p]
            ax.plot([xp, x], [yp-0.22, y+0.22], linewidth=1.4)
    ax.axis("off")
    if title: ax.set_title(title, fontsize=13, pad=8)

def draw_graph(ax, edges: List[Tuple[int,int]], title="Graph"):
    nodes = sorted(set([u for e in edges for u in e])); m = len(nodes)
    if m == 0:
        ax.text(0,0,"(no nodes)"); return
    R = 1.8; coords = {}
    for i,u in enumerate(nodes):
        angle = 2*np.pi*i/m; coords[u] = (R*np.cos(angle), R*np.sin(angle))
    for u,v in edges:
        x1,y1 = coords[u]; x2,y2 = coords[v]
        ax.plot([x1,x2],[y1,y2], linewidth=1.6)
    for u,(x,y) in coords.items():
        ax.add_patch(Circle((x,y), 0.22, fill=False, linewidth=2))
        ax.text(x, y, str(u), ha="center", va="center", fontsize=11)
    ax.axis("off")
    if title: ax.set_title(title, fontsize=13, pad=8)

def draw_queue(ax, items: List[str], title=""): draw_array(ax, items, title=title)
def draw_deque(ax, items_l: List[str], title=""):
    draw_array(ax, items_l, title=title)
    ax.text(0, 1.2, "Front ⟵ add/remove", fontsize=10)
    ax.text(len(items_l)*1.3, 1.2, "add/remove ⟶ Rear", fontsize=10, ha="right")
def draw_stack(ax, items: List[str], title=""):
    h = 0.8
    for i, v in enumerate(items):
        y = i*h
        ax.add_patch(Rectangle((0, y), 2.2, h, fill=False, linewidth=2))
        ax.text(1.1, y+h/2, str(v), ha="center", va="center", fontsize=12)
    ax.text(2.6, len(items)*h - h/2 if items else 0, "TOP →", va="center", fontsize=11)
    ax.set_xlim(-0.5, 3.1); ax.set_ylim(-0.5, max(3.5, len(items)*h + 0.5)); ax.axis("off")
    if title: ax.set_title(title, fontsize=13, pad=8)

# -------------------------------------------------------
# Quiz helpers
# -------------------------------------------------------


def quiz_block(questions: List[Dict[str, Any]], key_prefix: str):
    """Per-topic mini-quiz with per-question feedback."""
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}. {q['q']}**")
        choice = st.radio(
            f"{key_prefix}_q{i}",
            q["options"],
            key=f"{key_prefix}_q{i}",
            index=0,
            label_visibility="collapsed",
        )
        if st.button(f"Check Q{i+1}", key=f"{key_prefix}_btn{i}"):
            idx = q["options"].index(choice)
            if idx == q["answer"]:
                card("tip", "Correct ✅", q["explain"])
            else:
                card("warn", "Not quite ❌", q["explain"])

def quiz_block_with_score(questions: List[Dict[str, Any]], key_prefix: str, store_key: str):
    """Full 10Q bank scorer (Practice page)."""
    answers: Dict[int, int] = {}
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}. {q['q']}**")
        choice = st.radio(
            f"{key_prefix}_q{i}",
            q["options"],
            key=f"{key_prefix}_q{i}",
            index=0,
            label_visibility="collapsed",
        )
        answers[i] = q["options"].index(choice)
    if st.button("Check All", key=f"{key_prefix}_checkall"):
        score = 0
        for i, q in enumerate(questions):
            if answers[i] == q["answer"]:
                score += 1
                card("tip", f"Q{i+1} ✅", q["explain"])
            else:
                card("warn", f"Q{i+1} ❌", q["explain"])
        st.success(f"Score: {score} / {len(questions)}")
        st.session_state.setdefault("scores", {})[store_key] = score

# -------------------------------------------------------
# PRACTICE banks — exact 10Q sets
# -------------------------------------------------------
def bank_logic():
    return [
        {"q":"First step before coding:", "options":["Start coding","Clarify I/O & constraints","Benchmark","Refactor"], "answer":1, "explain":"Clarity avoids wrong paths."},
        {"q":"Tiny examples help because:", "options":["Faster typing","Reveal rules & edges","Guarantee AC","Skip math"], "answer":1, "explain":"They expose edge cases cheaply."},
        {"q":"Edge case NOT typical:", "options":["Empty input","Huge input","Random seed","Negative values"], "answer":2, "explain":"Random seed isn't a logic edge case."},
        {"q":"Good problem restatement:", "options":["Copy prompt","Summarize in your words","List libraries","Draw UML first"], "answer":1, "explain":"Own words ensure understanding."},
        {"q":"When to design tests:", "options":["After code","Before coding","Never","At deployment"], "answer":1, "explain":"Test-first reveals spec gaps."},
        {"q":"Algorithm is:", "options":["Source code","Ordered steps to solve","Runtime log","IDE setting"], "answer":1, "explain":"Steps independent of language."},
        {"q":"Common pitfall:", "options":["Check constraints","Plan for edges","Jump to code","Use examples"], "answer":2, "explain":"Skipping plan leads to bugs."},
        {"q":"Input/Output spec is:", "options":["Nice-to-have","Optional","Essential","Only for APIs"], "answer":2, "explain":"Defines correctness."},
        {"q":"State invariants:", "options":["Are magic","Always false","Help reason about steps","Only for math"], "answer":2, "explain":"Invariants guide correctness."},
        {"q":"Dry run means:", "options":["Unit test","Execute mentally on small data","Profiling","Refactor"], "answer":1, "explain":"Manually trace to validate logic."},
    ]

def bank_complexity():
    return [
        {"q":"Drop constants in Big-O because:", "options":["They’re zero","Don’t matter as n→∞","Tradition","Licensing"], "answer":1, "explain":"Asymptotics compare growth."},
        {"q":"Which grows slowest:", "options":["O(n)","O(log n)","O(n log n)","O(n²)"], "answer":1, "explain":"Logarithmic is slowest here."},
        {"q":"Binary search:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":1, "explain":"Halving strategy."},
        {"q":"Merge sort time:", "options":["O(n)","O(n log n)","O(n²)","O(log n)"], "answer":1, "explain":"Divide & conquer."},
        {"q":"Space: recursion depth of quicksort avg:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":1, "explain":"Average depth ~ log n."},
        {"q":"Hash average lookup:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":0, "explain":"With good hash & load factor."},
        {"q":"Tightest bound means:", "options":["Any bound","As precise as possible","Worst only","Best only"], "answer":1, "explain":"Theta is tight bound."},
        {"q":"Amortized O(1) example:", "options":["Vector push_back","Binary search","Merge sort","DFS"], "answer":0, "explain":"Occasional resize, average O(1)."},
        {"q":"Exponential class:", "options":["O(2^n)","O(n log n)","O(log n)","O(1)"], "answer":0, "explain":"Explodes with n."},
        {"q":"Polynomial vs logarithmic:", "options":["Poly slower for large n","Log slower","Same","Depends on base only"], "answer":0, "explain":"Any n^k outgrows log n."},
    ]

def bank_array():
    return [
        {"q":"Access arr[i]:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":0, "explain":"Direct offset."},
        {"q":"Insert at middle:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":2, "explain":"Shift elements."},
        {"q":"Append (amortized) in dynamic array:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":0, "explain":"Occasional resize."},
        {"q":"Best for:", "options":["Random access","Frequent mid inserts","Frequent deletions head","Graph cycles"], "answer":0, "explain":"Indexing is strength."},
        {"q":"Cache-friendly due to:", "options":["Contiguity","Pointers","Hashing","Trees"], "answer":0, "explain":"Sequential memory."},
        {"q":"Find x in unsorted:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":2, "explain":"Linear scan."},
        {"q":"Stable index range:", "options":["1..n","0..n-1","-1..n","None"], "answer":1, "explain":"Zero-based usually."},
        {"q":"Out-of-bounds:", "options":["Safe","Undefined/exception","Sorted","Hashed"], "answer":1, "explain":"Access error."},
        {"q":"Slicing cost in Python lists:", "options":["O(1) copy","O(k) to copy slice","O(n log n)","O(1) view"], "answer":1, "explain":"Copies elements."},
        {"q":"Grow by doubling gives:", "options":["Amortized O(1) push","O(n) push","O(log n) push","No effect"], "answer":0, "explain":"Resize rare."},
    ]

def bank_search():
    return [
        {"q":"Binary search needs:", "options":["Linked list","Sorted array","Hash map","Randomness"], "answer":1, "explain":"Order gives direction."},
        {"q":"Linear search worst case:", "options":["1","log n","n","n log n"], "answer":2, "explain":"Check all."},
        {"q":"Ternary search typical use:", "options":["Unimodal function","Hash table","Matrix","Deque"], "answer":0, "explain":"Find max/min on convex/concave."},
        {"q":"Interpolation search performs well when:", "options":["Uniformly distributed keys","Reverse-sorted","Random links","Trees"], "answer":0, "explain":"Probes near key location."},
        {"q":"Exponential search helps to:", "options":["Find bounds then binary search","Sort array","Hash values","Build heap"], "answer":0, "explain":"Find range quickly."},
        {"q":"Binary search mid update bug:", "options":["lo = mid+1/ hi = mid-1","Use floats","Check equals","Return -1"], "answer":0, "explain":"Must move bounds to avoid infinite loop."},
        {"q":"Search in rotated sorted:", "options":["Linear only","Modified binary search","Hashing","DP"], "answer":1, "explain":"Use sorted half property."},
        {"q":"If duplicates present:", "options":["Binary invalid","Still works; tweak for first/last","Must sort again","Use stack"], "answer":1, "explain":"Find boundary."},
        {"q":"Lower_bound returns:", "options":["Last < x","First ≥ x","First > x","Any x"], "answer":1, "explain":"Standard definition."},
        {"q":"Upper_bound returns:", "options":["Last ≤ x","First > x","First ≥ x","Any x"], "answer":1, "explain":"Strict greater."},
    ]

def bank_sorting():
    return [
        {"q":"Insertion sort avg:", "options":["O(n)","O(n log n)","O(n²)","O(log n)"], "answer":2, "explain":"Shifts elements."},
        {"q":"Merge sort is:", "options":["In-place","Stable (commonly)","Unstable always","Quadratic"], "answer":1, "explain":"Merging preserves order of equals."},
        {"q":"Quick sort avg time:", "options":["O(n)","O(n log n)","O(n²)","O(n!)"], "answer":1, "explain":"Partition-based D&C."},
        {"q":"Quick sort worst case occurs when:", "options":["Balanced pivots","All equal","Sorted with poor pivot","Random pivot"], "answer":2, "explain":"Skewed partitions."},
        {"q":"Heap sort time:", "options":["O(n)","O(n log n)","O(n²)","O(log n)"], "answer":1, "explain":"Build heap + n extract."},
        {"q":"Counting sort requires:", "options":["Comparable only","Small integer range","Hash table","Linked list"], "answer":1, "explain":"Counts occurrences."},
        {"q":"Stable means:", "options":["Preserves equal keys order","Fastest","In-place","Recursive"], "answer":0, "explain":"Relative order of equals kept."},
        {"q":"Best for nearly-sorted:", "options":["Insertion sort","Merge sort","Heap sort","Quick sort"], "answer":0, "explain":"Few shifts."},
        {"q":"TimSort used in Python sorts:", "options":["Yes","No"], "answer":0, "explain":"Hybrid of merge + insertion."},
        {"q":"External sorting used when:", "options":["RAM suffices","Data > memory","Few items","Linked list"], "answer":1, "explain":"Use disk runs & merges."},
    ]

def bank_hashing():
    return [
        {"q":"Hash table average lookup:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":0, "explain":"With good hash & load factor."},
        {"q":"Collision handling NOT typical:", "options":["Chaining","Open addressing","Counting sort","Cuckoo"], "answer":2, "explain":"Counting sort isn’t a collision method."},
        {"q":"Load factor α:", "options":["m/n","n/m","n·m","m-n"], "answer":1, "explain":"α = items/buckets."},
        {"q":"Primary clustering issue in:", "options":["Chaining","Linear probing","Separate store","BST"], "answer":1, "explain":"Runs form for linear probing."},
        {"q":"Double hashing reduces:", "options":["Keys","Probing length correlations","Buckets","Memory"], "answer":1, "explain":"Independent step size."},
        {"q":"Good m often:", "options":["Power of 2 only","Prime-ish","Always 10","Random"], "answer":1, "explain":"Spreads keys better."},
        {"q":"Rehashing triggers when:", "options":["α small","α large","Size prime","Keys sorted"], "answer":1, "explain":"Too crowded."},
        {"q":"Cryptographic hash is designed for:", "options":["Speed only","Strong collision resistance","Small memory","Sorting"], "answer":1, "explain":"Security properties."},
        {"q":"Perfect hashing means:", "options":["No collisions on set","Always no collisions","Sorted keys","Double tables"], "answer":0, "explain":"For a fixed key set."},
        {"q":"Bloom filter:", "options":["Exact set","Prob. set with FPs","BST","Deque"], "answer":1, "explain":"Space-efficient membership with false positives."},
    ]

def bank_twoptr():
    return [
        {"q":"Classic pair-sum needs:", "options":["Random order","Sorted array","Graph","Heap"], "answer":1, "explain":"Movement relies on order."},
        {"q":"If sum < target:", "options":["Move i++","Move j--","Stop","Reverse"], "answer":0, "explain":"Increase sum by moving left pointer right."},
        {"q":"If sum > target:", "options":["i++","j--","Stop","Swap i,j"], "answer":1, "explain":"Decrease sum by moving right pointer left."},
        {"q":"Remove duplicates (sorted) uses:", "options":["Two pointers","Stack","Queue","Hashset only"], "answer":0, "explain":"Slow build, fast scan."},
        {"q":"Trapping rain water often uses:", "options":["Two pointers","DFS","DP only","Sorting"], "answer":0, "explain":"Comparing left/right max."},
        {"q":"3-sum after sort often uses:", "options":["Two pointers inside loop","Hash for all","Pure DP","Segment tree"], "answer":0, "explain":"Fix one, two-pointer others."},
        {"q":"Move zeros to end with order:", "options":["Two pointers","Heap","Stack","Graph"], "answer":0, "explain":"Write pointer + scan."},
        {"q":"Min diff in sorted arrays:", "options":["Two pointers","Hash","Heap","Stack"], "answer":0, "explain":"Advance smaller side."},
        {"q":"Is palindrome (string) can be:", "options":["Two pointers","Hash only","Graph","Heap"], "answer":0, "explain":"Compare ends inward."},
        {"q":"Merge two sorted arrays efficiently:", "options":["Two pointers","Binary search each","Heapify","Sort union"], "answer":0, "explain":"Walk both lists once."},
    ]

def bank_window():
    return [
        {"q":"Fixed window keeps:", "options":["Recompute sum","Running sum","Sort each","Stack"], "answer":1, "explain":"Add new, subtract old."},
        {"q":"Variable window use-case:", "options":["Exact length only","At most K distinct chars","Tree DP","Heap sort"], "answer":1, "explain":"Expand/shrink to satisfy constraint."},
        {"q":"Max subarray sum size k:", "options":["Fixed window","Two pointers only","Binary search","DP only"], "answer":0, "explain":"O(n) with running sum."},
        {"q":"Longest substring without repeat:", "options":["Variable window + set/map","Stack","Queue","Greedy only"], "answer":0, "explain":"Slide while maintaining uniqueness."},
        {"q":"Min window substring:", "options":["Variable window + counts","Sort","Heap","Graph"], "answer":0, "explain":"Expand to cover, shrink to minimal."},
        {"q":"Window shift cost target:", "options":["O(n)","O(1)","O(log n)","O(n log n)"], "answer":1, "explain":"Constant-time update."},
        {"q":"Monotonic deque helps with:", "options":["Window maxima","Sorting","Graph match","Hash collision"], "answer":0, "explain":"Keep candidates in order."},
        {"q":"When k>n:", "options":["Window invalid","Still okay","Binary search","DP"], "answer":0, "explain":"No k-sized window exists."},
        {"q":"Counting vowels in each k window:", "options":["Recount fully","Slide counts","Hash by key","Stack"], "answer":1, "explain":"Update counts on slide."},
        {"q":"Product of window numbers careful with:", "options":["Zeros","Ones","Twos","Sorting"], "answer":0, "explain":"Reset or handle zero specially."},
    ]

def bank_prefix():
    return [
        {"q":"pref[i] equals:", "options":["arr[i]","sum arr[0..i]","arr[i-1]","i"], "answer":1, "explain":"Cumulative total."},
        {"q":"Sum(l..r) via prefix:", "options":["pref[l]-pref[r]","pref[r]-pref[l-1]","pref[r+1]","arr[r]-arr[l]"], "answer":1, "explain":"Standard formula."},
        {"q":"When l=0:", "options":["Use pref[-1]","Use pref[r]","Undefined","Add again"], "answer":1, "explain":"Sum(0..r)=pref[r]."},
        {"q":"Precompute cost:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":2, "explain":"Single pass."},
        {"q":"Many range queries favor:", "options":["Prefix sums","Binary search","Stacks","Greedy"], "answer":0, "explain":"O(1) per query."},
        {"q":"2D prefix sums answer:", "options":["Point queries only","Submatrix sums","Sorting","Graph paths"], "answer":1, "explain":"Inclusion-exclusion."},
        {"q":"Difference array supports:", "options":["Point updates fast","Range updates fast","Heap ops","Graph DP"], "answer":1, "explain":"Then prefix to recover final array."},
        {"q":"Prefix vs Fenwick/Segment tree:", "options":["Same always","Trees support updates efficiently","Prefix faster for updates","Trees slower queries"], "answer":1, "explain":"Fenwick/Segment handle updates + queries."},
        {"q":"Running average easy with:", "options":["Prefix sums","Stacks","Heaps","Graphs"], "answer":0, "explain":"(pref[r]-pref[l-1])/(r-l+1)."},
        {"q":"Use long type for:", "options":["Tiny numbers","Possible overflow","Strings","Graphs"], "answer":1, "explain":"Cumulative sums can overflow small ints."},
    ]

def bank_string():
    return [
        {"q":"Strings often are:", "options":["Mutable","Immutable","Graphs","Heaps"], "answer":1, "explain":"New copy on change."},
        {"q":"Index base usually:", "options":["1","0","OS dependent","Random"], "answer":1, "explain":"Zero-based common."},
        {"q":"Reverse string naive cost:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":2, "explain":"Touch each char."},
        {"q":"Anagram check uses:", "options":["Sort/count chars","Graph","Deque","Heap"], "answer":0, "explain":"Compare frequency."},
        {"q":"Palindrome check uses:", "options":["Two pointers","DP only","Heap","Graph"], "answer":0, "explain":"Ends inward."},
        {"q":"Substring search naive:", "options":["O(nm)","O(1)","O(n)","O(log n)"], "answer":0, "explain":"Check each start."},
        {"q":"KMP improves by:", "options":["Heuristics","Prefix function (lps)","Sorting","Hash table"], "answer":1, "explain":"Avoid re-checking."},
        {"q":"Case-folding matters for:", "options":["Equality only","Unicode-aware compare","Graph DP","Binary heap"], "answer":1, "explain":"Correct comparisons."},
        {"q":"Trim removes:", "options":["Middle spaces","Leading/trailing spaces","All spaces","No change"], "answer":1, "explain":"Whitespace ends."},
        {"q":"Split/join pair is:", "options":["Hashing","Tokenization","Graphing","Sorting"], "answer":1, "explain":"Turn string to list and back."},
    ]

def bank_recursion():
    return [
        {"q":"Every recursion needs:", "options":["Greedy","Base case","Heap","Queue"], "answer":1, "explain":"To stop descent."},
        {"q":"Stack overflow risk when:", "options":["Base case fast","No base/progress","Tail call","Memoized"], "answer":1, "explain":"Infinite recursion."},
        {"q":"Divide-and-conquer example:", "options":["Merge sort","Hashing","Queueing","Heapify only"], "answer":0, "explain":"Split-merge."},
        {"q":"Tree traversal often is:", "options":["Iterative only","Recursive","Hash-based","Greedy"], "answer":1, "explain":"Structure matches recursion."},
        {"q":"Memoization helps by:", "options":["Recomputing subproblems","Storing & reusing sub-results","Random restarts","Greedy picks"], "answer":1, "explain":"Cache results so overlapping subproblems aren’t recomputed."},
        {"q":"Tail recursion optimization can:", "options":["Increase depth","Remove need for base case","Turn some calls into loops","Sort faster"], "answer":2, "explain":"Some languages optimize tail calls, similar to loops."},
        {"q":"Backtracking typically:", "options":["Never branches","Explores choices & undoes (pop) on failure","Uses heaps only","Needs hashing"], "answer":1, "explain":"Try a choice, recurse, undo if it doesn’t work."},
        {"q":"Recurrence T(n)=2T(n/2)+O(n) solves to:", "options":["O(n)","O(n log n)","O(n²)","O(log n)"], "answer":1, "explain":"Master Theorem: a=b=2, f(n)=n ⇒ O(n log n)."},
        {"q":"Tree height (balanced) vs (skewed):", "options":["O(log n) vs O(n)","O(n) vs O(log n)","O(1) vs O(n)","Same"], "answer":0, "explain":"Balanced trees keep height logarithmic; skewed can be linear."},
        {"q":"Choose recursion when:", "options":["State fits naturally as smaller subproblems","Iteration impossible","Stack is unlimited","Big-O always better"], "answer":0, "explain":"Use when problem structure is self-similar (trees, D&C)."},
    ]

def bank_matrix():
    return [
        {"q":"Grid element access a[r][c] means:", "options":["a[c][r]","a[r][c]","a[r+c]","a[r*c]"], "answer":1, "explain":"Row index first, then column."},
        {"q":"Neighbors for 4-dir BFS:", "options":["N,E,S,W","Only diagonals","Only vertical","Only horizontal"], "answer":0, "explain":"Optionally add diagonals for 8-dir."},
        {"q":"Count islands uses:", "options":["BFS/DFS on grid","Sorting rows","Greedy coins","Stacks only"], "answer":0, "explain":"Traverse connected components of 1s."},
        {"q":"Out-of-bounds errors avoided by:", "options":["Random checks","Boundary guards on r,c","Hashing","Heaps"], "answer":1, "explain":"Check 0≤r<R and 0≤c<C."},
        {"q":"2D prefix sums answer:", "options":["Point updates only","Submatrix sums quickly","Shortest paths","Spanning trees"], "answer":1, "explain":"Use inclusion–exclusion on prefix table."},
        {"q":"Row-major layout stores:", "options":["Cols contiguous","Rows contiguous","Random","Diagonal contiguous"], "answer":1, "explain":"Rows placed one after another in memory."},
        {"q":"Spiral traversal is:", "options":["BFS","Layer-wise peel (simulate bounds)","Dijkstra","Toposort"], "answer":1, "explain":"Adjust top/bottom/left/right bounds."},
        {"q":"Matrix multiplication naive:", "options":["O(n)","O(n log n)","O(n³)","O(n²)"], "answer":2, "explain":"Triple loop for n×n."},
        {"q":"Path in maze minimal steps:", "options":["DFS","BFS","Greedy","Union-Find"], "answer":1, "explain":"Unweighted grid shortest path via BFS."},
        {"q":"Transpose swaps:", "options":["Rows with columns","Values randomly","Edges","Diagonals only"], "answer":0, "explain":"A[r][c] ↔ A[c][r]."},
    ]

def bank_ll():
    return [
        {"q":"Linked list access by index is:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":2, "explain":"Must traverse nodes."},
        {"q":"Singly list node has:", "options":["data,next","data,prev,next","data,left,right","only data"], "answer":0, "explain":"Next pointer only."},
        {"q":"Delete after given node:", "options":["O(1)","O(log n)","O(n)","O(n²)"], "answer":0, "explain":"If you have pointer to prior node."},
        {"q":"Reverse list iteratively uses:", "options":["2 pointers","3 pointers (prev,cur,nxt)","Stack only","Recursion only"], "answer":1, "explain":"Standard pattern."},
        {"q":"Find cycle uses:", "options":["Sorting","Floyd’s tortoise-hare","Stack","Heap"], "answer":1, "explain":"Two-speed pointers."},
        {"q":"DLL advantage:", "options":["O(1) random access","Easy backward traversal","Less memory","Cache-friendly"], "answer":1, "explain":"Prev pointer enables reverse walk."},
        {"q":"Middle of list:", "options":["Two pointers (slow/fast)","Hash","Sort","Deque"], "answer":0, "explain":"Fast advances by 2."},
        {"q":"Merge two sorted lists:", "options":["Two pointers","Binary search","Heapify","Sort union"], "answer":0, "explain":"Linear merge."},
        {"q":"Head removal in singly:", "options":["O(1)","O(log n)","O(n)","Undefined"], "answer":0, "explain":"Repoint head."},
        {"q":"Lru cache often backed by:", "options":["Deque","Hash + DLL","Heap","Set"], "answer":1, "explain":"O(1) move-to-front and lookup."},
    ]

def bank_stack():
    return [
        {"q":"Stack order:", "options":["FIFO","LIFO","Sorted","Random"], "answer":1, "explain":"Last in, first out."},
        {"q":"Valid parentheses uses:", "options":["Queue","Stack","Heap","Graph"], "answer":1, "explain":"Push opens, pop closes."},
        {"q":"Call stack tracks:", "options":["Function frames","Heap objects","Files","Threads"], "answer":0, "explain":"Return addresses, locals."},
        {"q":"Postfix evaluation uses:", "options":["Stack","Deque","Queue","Set"], "answer":0, "explain":"Push operands, pop on operator."},
        {"q":"Undo/Redo pattern:", "options":["Two stacks","One queue","Hashmap","Heap"], "answer":0, "explain":"Undo/Redo stacks."},
        {"q":"Stack overflow occurs when:", "options":["Too many recursive calls","Too many heap objects","Too many files","Too many threads"], "answer":0, "explain":"Call depth exceeds limit."},
        {"q":"Monotonic stack helps:", "options":["Next greater element","BFS","Dijkstra","Union-find"], "answer":0, "explain":"Keep indices in order."},
        {"q":"Push/pop complexity:", "options":["O(1) avg","O(n)","O(log n)","O(n log n)"], "answer":0, "explain":"Constant time."},
        {"q":"Top (peek) is:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":0, "explain":"Direct access."},
        {"q":"Infix → postfix uses:", "options":["Operator stack","Queue only","Heap","Hash"], "answer":0, "explain":"Shunting-yard algorithm."},
    ]

def bank_queue():
    return [
        {"q":"Queue order:", "options":["FIFO","LIFO","Random","Sorted"], "answer":0, "explain":"First in, first out."},
        {"q":"BFS uses:", "options":["Stack","Queue","Heap","Set"], "answer":1, "explain":"Level by level."},
        {"q":"Circular array queue avoids:", "options":["Indices","Shifting elements","Dequeues","Enqueues"], "answer":1, "explain":"Use head/tail modulo n."},
        {"q":"Enqueue/Dequeue cost:", "options":["O(1) avg","O(n)","O(log n)","O(n log n)"], "answer":0, "explain":"Constant at ends."},
        {"q":"Priority queue differs:", "options":["FIFO by time","By priority key","Random","Round-robin"], "answer":1, "explain":"Order by priority."},
        {"q":"Blocking queues used in:", "options":["Concurrency/producer-consumer","Sorting","Hashing","Graphs"], "answer":0, "explain":"Thread handoff."},
        {"q":"Two-stack queue gives amortized:", "options":["O(1) ops","O(n) ops","O(log n) ops","Undefined"], "answer":0, "explain":"Move only when needed."},
        {"q":"Front access:", "options":["O(1)","O(n)","O(log n)","Depends"], "answer":0, "explain":"Pointer to head."},
        {"q":"Deque differs since:", "options":["Both ends ops","Random access","Sorted always","Uses hashing"], "answer":0, "explain":"Double ended."},
        {"q":"Queue best for:", "options":["Depth-first","Breadth-first","Sorting","DP"], "answer":1, "explain":"Breadth-wise expansion."},
    ]

def bank_deque():
    return [
        {"q":"Deque supports:", "options":["Front ops","Back ops","Both","Neither"], "answer":2, "explain":"Double-ended operations."},
        {"q":"Window max often uses:", "options":["Monotonic deque","Stack","Heap only","Graph"], "answer":0, "explain":"Maintain decreasing indices."},
        {"q":"Palindrome check can use deque by:", "options":["Pop both ends","Hashing","Heap","DP"], "answer":0, "explain":"Compare ends."},
        {"q":"Amortized cost per op:", "options":["O(1)","O(n)","O(log n)","O(n log n)"], "answer":0, "explain":"Pointer updates."},
        {"q":"Deque vs list front insert:", "options":["Deque O(1) typical","List O(1) typical","Both O(n)","Both O(log n)"], "answer":0, "explain":"Deque optimized for ends."},
        {"q":"Implement with:", "options":["Doubly list or circular buffer","BST","Hash","Graph"], "answer":0, "explain":"Common backing."},
        {"q":"Use-case NOT typical:", "options":["Scheduling","Undo/redo","Random index access","Sliding windows"], "answer":2, "explain":"Deque isn’t for random access."},
        {"q":"Rotate is:", "options":["Move ends cyclically","Sort","Hash","DP"], "answer":0, "explain":"Shift positions circularly."},
        {"q":"appendleft/popleft exist in:", "options":["Python collections.deque","list","set","dict"], "answer":0, "explain":"Deque API."},
        {"q":"Space overhead vs list:", "options":["Often higher","Always lower","Equal","Undefined"], "answer":0, "explain":"Extra pointers/buffer mgmt."},
    ]

def bank_tree():
    return [
        {"q":"Tree is:", "options":["Acyclic connected graph","Cyclic","DAG only","List"], "answer":0, "explain":"No cycles."},
        {"q":"Inorder of BST gives:", "options":["Random","Sorted","Reverse-sorted","Levels"], "answer":1, "explain":"BST property."},
        {"q":"Height of single node:", "options":["0 or 1 depending on def","n","-1 always","∞"], "answer":0, "explain":"Common: height(root)=0 or 1 by convention."},
                {"q":"Level-order traversal uses:", "options":["Stack","Queue","Heap","Set"], "answer":1, "explain":"BFS over tree uses a queue."},
        {"q":"BST search average time:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":1, "explain":"Balanced BST height ~ log n."},
        {"q":"BST worst case (skewed):", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":2, "explain":"Height can be n."},
        {"q":"Balanced BST example:", "options":["AVL/Red-Black","Linked list","Heap","Array"], "answer":0, "explain":"Self-balancing keep height logarithmic."},
        {"q":"Full vs complete tree:", "options":["Full: all nodes 2 or 0 kids; Complete: filled levels left-to-right","Same","Opposites","Only for BSTs"], "answer":0, "explain":"Standard definitions."},
        {"q":"Binary heap is:", "options":["Binary search tree","Complete binary tree","Arbitrary graph","Linked list"], "answer":1, "explain":"Heap is complete binary tree."},
        {"q":"Postorder useful for:", "options":["Evaluate expression trees","BFS","Hashing","Toposort"], "answer":0, "explain":"Children before parent."},
        {"q":"Diameter of tree:", "options":["Nodes count","Edges on longest path","Leaves count","Height of root"], "answer":1, "explain":"Longest path between two nodes."},
        {"q":"Serialize/deserialize commonly via:", "options":["Pre/level-order with null markers","Sorting keys","Hash","Heap"], "answer":0, "explain":"Traverse + placeholders reconstructs structure."},
    ]

def bank_heap():
    return [
        {"q":"Min-heap property:", "options":["Parent ≥ children","Parent ≤ children","Sorted in-order","Complete search tree"], "answer":1, "explain":"Each parent not greater than children."},
        {"q":"Heap find-min / find-max cost (min-heap):", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":0, "explain":"Min is at root."},
        {"q":"Insert/extract in binary heap:", "options":["O(1)","O(log n)","O(n)","Amortized O(n)"], "answer":1, "explain":"Bubble up / down along height ~ log n."},
        {"q":"Build-heap from array:", "options":["O(n)","O(n log n)","O(log n)","O(n²)"], "answer":0, "explain":"Floyd’s heapify bottom-up is linear."},
        {"q":"k largest from n (n≫k):", "options":["Max-heap of n","Min-heap of size k","Sort all","Queue"], "answer":1, "explain":"Keep k best; pop when size>k."},
        {"q":"Heapsort main steps:", "options":["Build + repeated extract","Merge runs","Count keys","Quick partition"], "answer":0, "explain":"Use heap to select next element."},
        {"q":"Priority queue backed by:", "options":["Stack","Heap","Hash","BST only"], "answer":1, "explain":"Heap supports priority ops."},
        {"q":"Heaps guarantee:", "options":["Global sort","Only top priority order","Stable order","Graph shortest paths"], "answer":1, "explain":"Only parent-child relation, not full order."},
        {"q":"D-ary heap advantage:", "options":["Cheaper decrease-key on all","Fewer levels (shallower)","No comparisons","Space-free"], "answer":1, "explain":"Higher branching reduces height."},
        {"q":"Index of children in array heap (0-based):", "options":["i-1 & i+1","2i+1 & 2i+2","i/2","i*i"], "answer":1, "explain":"Standard mapping."},
    ]

def bank_graph():
    return [
        {"q":"Good for sparse graph:", "options":["Adjacency matrix","Adjacency list","Heap","Stack"], "answer":1, "explain":"List stores only existing edges."},
        {"q":"BFS finds shortest paths in:", "options":["Weighted graphs","Unweighted graphs","Only DAGs","Only trees"], "answer":1, "explain":"Min edges count with BFS."},
        {"q":"Dijkstra requires:", "options":["Negative edges ok","Non-negative weights","Directed only","Tree only"], "answer":1, "explain":"Fails with negative edges (without tweaks)."},
        {"q":"Detect cycle in directed graph:", "options":["BFS only","DFS with colors/rec stack","Sort","Greedy"], "answer":1, "explain":"Back-edge indicates a cycle."},
        {"q":"Topological sort applies to:", "options":["Any graph","DAG","Trees only","Bipartite only"], "answer":1, "explain":"Linear order where edges go forward."},
        {"q":"MST algorithms:", "options":["Kruskal/Prim","Dijkstra/Bellman","KMP/Trie","AVL/Red-Black"], "answer":0, "explain":"Minimum spanning tree."},
        {"q":"Union-Find used in:", "options":["Kruskal’s MST","BFS","KMP","Heap sort"], "answer":0, "explain":"Detects cycles when adding edges."},
        {"q":"Bipartite check:", "options":["Color with 3 colors","2-color via BFS/DFS","Sort vertices","Use heap"], "answer":1, "explain":"Alternate levels; odd cycle breaks bipartite."},
        {"q":"Connected components (undirected):", "options":["Repeated BFS/DFS","Toposort","Dijkstra","Heapify"], "answer":0, "explain":"Traverse from each unvisited node."},
        {"q":"Floyd–Warshall solves:", "options":["Single-source shortest path","All-pairs shortest paths","MST","Max flow"], "answer":1, "explain":"Dynamic programming over intermediates."},
    ]

def bank_greedy():
    return [
        {"q":"Greedy optimal when:", "options":["We test 2 inputs","Greedy-choice & optimal substructure hold","It’s fastest","Recursion present"], "answer":1, "explain":"Need property/proof."},
        {"q":"Activity selection strategy:", "options":["Earliest start","Shortest duration","Earliest finish times","Random"], "answer":2, "explain":"Pick earliest finishing compatible activity."},
        {"q":"Fractional knapsack greedy key:", "options":["Value","Weight","Value/weight ratio","Index"], "answer":2, "explain":"Take highest ratio first; fractional allowed."},
        {"q":"Huffman coding builds:", "options":["Balanced BST","Prefix-free optimal codes","Hash","Deque"], "answer":1, "explain":"Merge two least frequent repeatedly."},
        {"q":"Interval scheduling maximum non-overlap:", "options":["Sort by start","Sort by finish","Sort by length","No sort"], "answer":1, "explain":"Earliest finish works."},
        {"q":"Coin change greedy works in:", "options":["Any coin set","Canonical systems (e.g., US/INR)","Never","Only primes"], "answer":1, "explain":"Fails for some denominations."},
        {"q":"Kruskal’s algorithm is:", "options":["Dynamic programming","Greedy","Divide & conquer","Backtracking"], "answer":1, "explain":"Add lowest edges without cycles."},
        {"q":"Minimum platforms / rooms problem:", "options":["Greedy with timeline sweep","DP only","Stack","Hash"], "answer":0, "explain":"Sort start/end; track concurrent count."},
        {"q":"Job sequencing with deadlines (profit):", "options":["Sort by deadline","Sort by profit desc + slot from end","Random","DP only"], "answer":1, "explain":"Greedy place most profitable first."},
        {"q":"Greedy fails when:", "options":["Local ≠ global optimum","All weights equal","Inputs small","We use sets"], "answer":0, "explain":"Counterexamples break greedy."},
    ]


# -------------------------------------------------------
# Live Demos: one-click, no-code runnable examples/topic
# -------------------------------------------------------
@dataclass
class Demo:
    title: str
    code: str                   # pretty-printed code we’ll show
    runner: Any                 # a function that runs and prints/plots
    explainer: str              # concise explanation of what’s happening

def _print_result(label, value):
    st.markdown(f"**{label}:** {value}")

def _show_code(code_text: str):
    st.code(dedent(code_text), language="python")

# --- All demo runner functions (small, deterministic, fast) ---
def run_logic():
    task = "Sum positives then count negatives"
    data = [3, -1, 5, -7, 2, 0, -4]
    s = sum(x for x in data if x > 0)
    c = sum(1 for x in data if x < 0)
    _print_result("Task", task)
    _print_result("Data", data)
    _print_result("Sum of positives", s)
    _print_result("Count of negatives", c)

def run_complexity():
    n = 2_000
    ops_lin = n
    ops_nlogn = int(n * math.log2(n))
    ops_quad = n * n
    _print_result("n", n)
    _print_result("~O(n) operations", ops_lin)
    _print_result("~O(n log n) operations", ops_nlogn)
    _print_result("~O(n²) operations", ops_quad)
    st.caption("Counts illustrate growth only (not timing).")

def run_array():
    arr = [7,2,9,3,1]
    _print_result("Array", arr)
    _print_result("arr[2]", arr[2])
    i, x = 2, 99
    arr2 = arr[:i] + [x] + arr[i:]   # simulate O(n) insert in middle
    _print_result("Insert 99 @ index 2", arr2)

def run_search():
    def binary_search(a, target):
        lo, hi = 0, len(a)-1
        while lo <= hi:
            mid = (lo+hi)//2
            if a[mid] == target: return mid
            if a[mid] < target: lo = mid+1
            else: hi = mid-1
        return -1
    a = [3,6,9,12,15,22,30]
    target = 22
    idx = binary_search(a, target)
    _print_result("Array", a)
    _print_result("Target", target)
    _print_result("Index found", idx)

def run_sorting():
    a = [7,2,9,3,1,3]
    sorted_a = sorted(a)  # TimSort (stable)
    _print_result("Original", a)
    _print_result("Sorted", sorted_a)
    st.caption("Python uses TimSort (merge+insertion; stable).")

def run_hashing():
    words = "to be or not to be that is the question".split()
    freq = defaultdict(int)
    for w in words: freq[w] += 1
    _print_result("Words", words)
    _print_result("Frequencies", dict(freq))

def run_twoptr():
    a = [1,2,3,7,8,12]
    target = 10
    i, j = 0, len(a)-1
    ans = None
    while i < j:
        s = a[i]+a[j]
        if s == target:
            ans = (a[i], a[j]); break
        if s < target: i += 1
        else: j -= 1
    _print_result("Array", a)
    _print_result("Target", target)
    _print_result("Pair", ans)

def run_window():
    a = [4,2,1,7,8,1,2,8,1,0]; k = 3
    cur = sum(a[:k]); best = cur; idx = 0
    for r in range(k, len(a)):
        cur += a[r] - a[r-k]
        if cur > best: best, idx = cur, r-k+1
    _print_result("Array", a)
    _print_result("k", k)
    _print_result("Max sum window", (best, a[idx:idx+k], f"indices {idx}..{idx+k-1}"))

def run_prefix():
    a = [2,1,3,4,2,1]
    pref = np.cumsum(a).tolist()
    l, r = 2, 4
    rng = pref[r] - (pref[l-1] if l>0 else 0)
    _print_result("Array", a)
    _print_result("Prefix", pref)
    _print_result("Sum(2..4)", rng)

def run_string():
    s1, s2 = "listen", "silent"
    def anagram(x,y): return sorted(x)==sorted(y)
    _print_result("s1", s1); _print_result("s2", s2)
    _print_result("Are anagrams?", anagram(s1,s2))

def run_recursion():
    trace = []
    def fact(n):
        trace.append(f"fact({n})")
        return 1 if n==0 else n*fact(n-1)
    n=5; val=fact(n)
    _print_result("Trace", " → ".join(trace))
    _print_result("fact(5)", val)

def run_matrix():
    grid = [
        [1,1,1,0],
        [0,1,0,1],
        [0,1,1,1],
        [0,0,0,1],
    ]
    R,C=len(grid),len(grid[0])
    sr,sc,er,ec = (0,0, 3,3)
    q = deque([(sr,sc,0)]); vis={(sr,sc)}
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    dist=-1
    while q:
        r,c,d=q.popleft()
        if (r,c)==(er,ec): dist=d; break
        for dr,dc in dirs:
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc]==1 and (nr,nc) not in vis:
                vis.add((nr,nc)); q.append((nr,nc,d+1))
    _print_result("Grid (1=free,0=wall)", grid)
    _print_result("Start→End", ((sr,sc),(er,ec)))
    _print_result("Shortest steps (BFS)", dist)

def run_ll():
    class Node:
        def __init__(self,v,next=None): self.v=v; self.next=next
    def to_list(h):
        out=[]; 
        while h: out.append(h.v); h=h.next
        return out
    # build 10→20→30
    h=Node(10,Node(20,Node(30)))
    _print_result("Original", to_list(h))
    # reverse
    prev=None; cur=h
    while cur:
        nxt=cur.next; cur.next=prev; prev=cur; cur=nxt
    h=prev
    _print_result("Reversed", to_list(h))

def run_stack():
    s="(a+b)*{c+[d-(e/f)]}"
    mp={')':'(',']':'[','}':'{'}; stck=[]
    ok=True
    for ch in s:
        if ch in "([{": stck.append(ch)
        elif ch in ")]}":
            if not stck or stck.pop()!=mp[ch]: ok=False; break
    ok = ok and not stck
    _print_result("Expr", s)
    _print_result("Balanced?", ok)

def run_queue():
    # level order (BFS) on simple tree
    tree = {1:[2,3], 2:[4,5], 3:[6], 4:[],5:[],6:[]}
    q=deque([1]); order=[]
    while q:
        u=q.popleft(); order.append(u); q.extend(tree[u])
    _print_result("Tree adj", tree)
    _print_result("Level-order", order)

def run_deque():
    a=[1,3,-1,-3,5,3,6,7]; k=3
    dq=deque(); res=[]
    for i,x in enumerate(a):
        while dq and dq[0]<=i-k: dq.popleft()
        while dq and a[dq[-1]]<=x: dq.pop()
        dq.append(i)
        if i>=k-1: res.append(a[dq[0]])
    _print_result("Array", a)
    _print_result("k", k)
    _print_result("Window maxima", res)

def run_tree():
    # inorder traversal of BST built from list
    class T:
        __slots__=("v","l","r")
        def __init__(self,v): self.v=v; self.l=self.r=None
    def insert(root,x):
        if not root: return T(x)
        if x<root.v: root.l=insert(root.l,x)
        else: root.r=insert(root.r,x)
        return root
    def inorder(n, out):
        if not n: return
        inorder(n.l,out); out.append(n.v); inorder(n.r,out)
    vals=[7,3,9,1,5,10]
    root=None
    for v in vals: root=insert(root,v)
    out=[]; inorder(root,out)
    _print_result("Inserted", vals)
    _print_result("Inorder (sorted)", out)

def run_heap():
    a=[7,2,9,3,1,8,4]
    heap = a[:] ; heapq.heapify(heap)
    k=3; smallest=[heapq.heappop(heap) for _ in range(k)]
    _print_result("Array", a)
    _print_result("k", k)
    _print_result("k smallest", smallest)

def run_graph():
    # BFS shortest path length in unweighted graph
    g = {0:[1,2], 1:[2,3], 2:[3], 3:[4], 4:[]}
    s,t = 0,4
    q=deque([(s,0)]); seen={s}; dist=-1
    while q:
        u,d=q.popleft()
        if u==t: dist=d; break
        for v in g[u]:
            if v not in seen: seen.add(v); q.append((v,d+1))
    _print_result("Graph", g)
    _print_result("Source→Target", (s,t))
    _print_result("Shortest edges", dist)

def run_greedy():
    # activity selection by earliest finish
    acts = [(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(2,14),(12,16)]
    acts.sort(key=lambda x:x[1])
    res=[]; cur_end=-1
    for s,e in acts:
        if s>=cur_end: res.append((s,e)); cur_end=e
    _print_result("Activities (start,end)", acts)
    _print_result("Chosen (max non-overlap)", res)

# --- Demos registry (one entry per sidebar topic key) ---
DEMOS: Dict[str, Demo] = {
    "logic": Demo(
        "Build logic from spec",
        """
        data = [3, -1, 5, -7, 2, 0, -4]
        sum_pos = sum(x for x in data if x > 0)
        cnt_neg = sum(1 for x in data if x < 0)
        """,
        run_logic,
        "We convert English spec into simple loops/filters: one pass for sum of positives and one for count of negatives."
    ),
    "complexity": Demo(
        "Operation growth sketch",
        """
        n = 2000
        ops_lin = n
        ops_nlogn = int(n * math.log2(n))
        ops_quad = n * n
        """,
        run_complexity,
        "Shows how operation counts scale across O(n), O(n log n), and O(n²)."
    ),
    "array": Demo(
        "Indexing & mid-insert",
        """
        arr = [7,2,9,3,1]
        x = arr[2]            # O(1)
        arr2 = arr[:2] + [99] + arr[2:]   # O(n) mid insert
        """,
        run_array,
        "Reads are O(1); inserting mid shifts elements (O(n))."
    ),
    "search": Demo(
        "Binary Search",
        """
        def binary_search(a, target):
            lo, hi = 0, len(a)-1
            while lo <= hi:
                mid = (lo+hi)//2
                if a[mid] == target: return mid
                if a[mid] < target: lo = mid+1
                else: hi = mid-1
            return -1
        """,
        run_search,
        "On a sorted array, halve the search space each step → O(log n)."
    ),
    "sorting": Demo(
        "TimSort (Python sorted)",
        """
        arr = [7,2,9,3,1,3]
        sorted_arr = sorted(arr)   # stable
        """,
        run_sorting,
        "TimSort blends merge & insertion; stability preserves equal-key order."
    ),
    "hashing": Demo(
        "Word frequency via dict",
        """
        from collections import defaultdict
        freq = defaultdict(int)
        for w in words: freq[w] += 1
        """,
        run_hashing,
        "Hash map buckets give near O(1) average updates/lookups."
    ),
    "twoptr": Demo(
        "Pair sum on sorted",
        """
        i, j = 0, len(a)-1
        while i<j:
            s = a[i]+a[j]
            if s==target: break
            if s<target: i+=1
            else: j-=1
        """,
        run_twoptr,
        "Pointers move inward based on sum vs target."
    ),
    "window": Demo(
        "Max sum window (fixed k)",
        """
        cur = sum(a[:k]); best=cur; idx=0
        for r in range(k, len(a)):
            cur += a[r] - a[r-k]
            if cur>best: best, idx = cur, r-k+1
        """,
        run_window,
        "Maintain running sum: add right, subtract left → O(n)."
    ),
    "prefix": Demo(
        "Range sum via prefix",
        """
        pref = np.cumsum(a).tolist()
        rng = pref[r] - (pref[l-1] if l>0 else 0)
        """,
        run_prefix,
        "Precompute once; answer any range in O(1)."
    ),
    "string": Demo(
        "Anagram check",
        """
        def anagram(x,y): return sorted(x)==sorted(y)
        """,
        run_string,
        "Sort both; equal implies same multiset of chars."
    ),
    "recursion": Demo(
        "Factorial with trace",
        """
        def fact(n):
            return 1 if n==0 else n*fact(n-1)
        """,
        run_recursion,
        "Base case n=0; each call reduces n → call stack unwinds with products."
    ),
    "matrix": Demo(
        "BFS shortest path in grid",
        """
        q = deque([(sr,sc,0)]); vis={(sr,sc)}
        while q:
            r,c,d=q.popleft()
            if (r,c)==(er,ec): return d
            for nr,nc in neighbors:
                ...
        """,
        run_matrix,
        "Unweighted grids use BFS for minimal steps."
    ),
    "ll": Demo(
        "Reverse a singly linked list",
        """
        prev=None; cur=head
        while cur:
            nxt=cur.next; cur.next=prev
            prev=cur; cur=nxt
        head=prev
        """,
        run_ll,
        "Three-pointer pattern (prev, cur, next)."
    ),
    "stack": Demo(
        "Valid parentheses",
        """
        for ch in s:
            if ch in '([{': push
            elif ch in ')]}': pop and match
        """,
        run_stack,
        "LIFO stack pairs openers with closers."
    ),
    "queue": Demo(
        "Level-order traversal",
        """
        q=deque([root])
        while q: u=q.popleft(); enqueue children
        """,
        run_queue,
        "BFS visits nodes level by level using a queue."
    ),
    "deque": Demo(
        "Sliding window maxima",
        """
        while dq and dq[0] <= i-k: dq.popleft()
        while dq and a[dq[-1]] <= x: dq.pop()
        dq.append(i)
        """,
        run_deque,
        "Monotonic deque keeps candidate maxima indices."
    ),
    "tree": Demo(
        "BST inorder = sorted",
        """
        def inorder(n): inorder(n.l); visit(n); inorder(n.r)
        """,
        run_tree,
        "BST property + inorder yields ascending order."
    ),
    "heap": Demo(
        "k smallest via heapq",
        """
        heap = arr[:]; heapq.heapify(heap)
        smallest = [heapq.heappop(heap) for _ in range(k)]
        """,
        run_heap,
        "Binary heap supports O(log n) pushes/pops; O(n) heapify."
    ),
    "graph": Demo(
        "BFS shortest edges",
        """
        q=deque([(s,0)]); seen={s}
        while q:
            u,d=q.popleft()
            if u==t: return d
            for v in g[u]: if v not in seen: ...
        """,
        run_graph,
        "Unweighted shortest path length via BFS."
    ),
    "greedy": Demo(
        "Activity selection (earliest finish)",
        """
        acts.sort(key=lambda x:x[1])
        cur_end=-1
        for s,e in acts:
            if s>=cur_end: choose; cur_end=e
        """,
        run_greedy,
        "Greedy choice: always pick the activity finishing earliest that fits."
    ),
}

# ---------------------------
# Map topic->question bank
# ---------------------------
TOPIC_BANKS: Dict[str, Any] = {
    "logic": bank_logic, "complexity": bank_complexity, "array": bank_array,
    "search": bank_search, "sorting": bank_sorting, "hashing": bank_hashing,
    "twoptr": bank_twoptr, "window": bank_window, "prefix": bank_prefix,
    "string": bank_string, "recursion": bank_recursion, "matrix": bank_matrix,
    "ll": bank_ll, "stack": bank_stack, "queue": bank_queue, "deque": bank_deque,
    "tree": bank_tree, "heap": bank_heap, "graph": bank_graph, "greedy": bank_greedy,
}

def get_topic_bank(topic_key: str) -> List[Dict[str, Any]]:
    f = TOPIC_BANKS.get(topic_key)
    return f() if f else []

def render_topic_practice(topic_key: str, store_key: str):
    qs = get_topic_bank(topic_key)
    if not qs:
        st.info("No questions found for this topic yet.")
        return
    st.subheader("Practice: 10 Questions")
    quiz_block_with_score(qs, f"bank_{topic_key}", store_key)

# -------------------------------------------------------
# LEARN MODE topics (content + mini-quiz)
# -------------------------------------------------------
Topic = Dict[str, Any]
topics: List[Topic] = [
    {
     "key":"logic","title":"Logic Building",
     "what":"Turning vague problems into precise steps a computer can follow.",
     "why":"Clear logic prevents bugs and rework.",
     "analogy":"Recipe: inputs (ingredients) → steps (algorithm) → output (dish).",
     "how":[ "Restate the problem in your own words.","List inputs/outputs/constraints.",
             "Work a tiny example by hand.","Check edge cases.","Only then translate to code." ],
     "example":"Coins 10,5,1 to make 28 → 10,10,5,1,1,1.",
     "pitfalls":[ "Jumping to code without clarity", "Ignoring edge cases", "No hand-run on small example"],
     "quiz":[
       {"q":"First step in logic building?", "options":["Start coding ASAP","Clarify I/O & constraints","Memorize syntax","Search internet"], "answer":1, "explain":"Clarity first."},
       {"q":"Why tiny examples?", "options":["Faster typing","Expose rules & edges","Guarantee AC","Avoid math"], "answer":1, "explain":"Cheap way to validate steps."}
     ]
    },
    {
     "key":"complexity","title":"Learn about Complexities",
     "what":"Estimate time/space with input n using Big-O (usually worst case).",
     "why":"Predict performance and choose the right approach.",
     "analogy":"Traffic: O(1) empty road; O(n) steady; O(n²) jam.",
     "how":[ "Count dominant operations vs n.","Drop constants/lower terms.",
             "Know order: O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n) < O(n!)" ],
     "example":"Binary search halves each step → O(log n).",
     "pitfalls":[ "Mixing average/worst/best", "Keeping constants while comparing" ],
     "quiz":[
       {"q":"Which grows slowest?", "options":["O(n)","O(log n)","O(n log n)","O(n²)"], "answer":1, "explain":"Logarithmic is smallest here."},
       {"q":"Drop constants because…", "options":["They’re zero","As n→∞ they don’t matter","Tradition","Licensing"], "answer":1, "explain":"Asymptotics care about growth."}
     ]
    },
    {
     "key":"array","title":"Array",
     "what":"Contiguous memory slots indexed 0..n−1.",
     "why":"O(1) random access and cache locality.",
     "analogy":"Numbered lockers in a row.",
     "how":[ "Access O(1).","Mid insert/delete shifts O(n).","Search O(n) unless sorted." ],
     "example":"marks[1] is constant-time.",
     "pitfalls":[ "Off-by-one", "Forgetting mid operations are O(n)" ],
     "quiz":[
       {"q":"Access arr[i] is:", "options":["O(1)","O(n)","O(log n)","O(n²)"], "answer":0, "explain":"Direct offset."},
       {"q":"Insert at index 2:", "options":["O(1)","O(n)","O(log n)","O(n log n)"], "answer":1, "explain":"Shift right side."}
     ]
    },
    {
     "key":"search","title":"Searching Algorithms",
     "what":"Find item/position in data.",
     "why":"Core operation everywhere.",
     "analogy":"Phonebook vs random pile.",
     "how":[ "Linear for any order.","Binary on sorted.","Hash for O(1) average." ],
     "example":"Find 22 in [3,6,9,12,22].",
     "pitfalls":[ "Binary on unsorted", "Forgetting equality check" ],
     "quiz":[
       {"q":"Binary search requires:", "options":["Linked list","Sorted array","Hash","Randomness"], "answer":1, "explain":"Order drives direction."},
       {"q":"Linear worst case:", "options":["1","log n","n","n log n"], "answer":2, "explain":"May check all."}
     ]
    },
    {
     "key":"sorting","title":"Sorting Algorithms",
     "what":"Arrange in order.",
     "why":"Makes search/merge/dedup faster.",
     "analogy":"Roll numbers in ascending order.",
     "how":[ "Simple O(n²): Bubble/Selection/Insertion.","Efficient ~O(n log n): Merge/Quick/Heap.","Stability matters for equal keys." ],
     "example":"Alphabetical attendance.",
     "pitfalls":[ "Confusing quicksort avg vs worst", "Ignoring stability when needed" ],
     "quiz":[
       {"q":"Merge sort time:", "options":["O(n)","O(n log n)","O(n²)","O(log n)"], "answer":1, "explain":"Divide & merge."},
       {"q":"Stable usually:", "options":["Merge","Quick","Heap","Selection"], "answer":0, "explain":"Common implementation is stable."}
     ]
    },
    {
     "key":"hashing","title":"Hashing",
     "what":"Map keys → buckets by hash(key)%m.",
     "why":"Near O(1) average find/insert/delete.",
     "analogy":"Pigeonholes by code.",
     "how":[ "Choose good hash & m.","Handle collisions (chaining/open addressing).","Resize by load factor." ],
     "example":"Check if username taken.",
     "pitfalls":[ "Poor hash clustering", "Ignoring load factor" ],
     "quiz":[
       {"q":"Average lookup:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":0, "explain":"With good spread."},
       {"q":"Collision handling:", "options":["Chaining","Binary search","Counting sort","DFS"], "answer":0, "explain":"List per bucket."}
     ]
    },
    {
     "key":"twoptr","title":"Two Pointer Technique",
     "what":"Two indices moving with rules (often sorted arrays).",
     "why":"Turns nested loops to O(n).",
     "analogy":"Readers from both ends of a shelf.",
     "how":[ "Pair-sum on sorted: move i/j by comparing sum.","De-dup: slow builds, fast scans." ],
     "example":"Any pair sums to 10?",
     "pitfalls":[ "Needs sorted data for classic pattern" ],
     "quiz":[
       {"q":"Pair-sum needs:", "options":["Sorted","Random","Graph","Heap"], "answer":0, "explain":"Movement relies on order."},
       {"q":"If sum<target:", "options":["i++","j--","Stop","Reverse"], "answer":0, "explain":"Increase sum."}
     ]
    },
    {
     "key":"window","title":"Window Sliding Technique",
     "what":"Maintain subarray and update in O(1) per slide.",
     "why":"Reduce O(n·k) to O(n).",
     "analogy":"Magnifying glass sliding along text.",
     "how":[ "Fixed size: add right, remove left.","Variable: expand to satisfy, shrink to minimal." ],
     "example":"Max sum of 3 consecutive days.",
     "pitfalls":[ "Recompute full sum instead of updating" ],
     "quiz":[
       {"q":"Fixed window trick:", "options":["Recompute","Running sum","Sort","Recursion"], "answer":1, "explain":"Add new, subtract old."},
       {"q":"Variable window for:", "options":["Sum fixed k","Unique chars constraint","Matrix only","Trees only"], "answer":1, "explain":"Expand/shrink by rule."}
     ]
    },
    {
     "key":"prefix","title":"Prefix Sum Technique",
     "what":"Precompute cumulative sums to answer ranges in O(1).",
     "why":"Avoid repeat summing.",
     "analogy":"Running bank balance.",
     "how":[ "pref[i]=sum(0..i).","Sum(l..r)=pref[r]-pref[l-1].","Extend to 2D." ],
     "example":"Rainfall days 10..20 quickly.",
     "pitfalls":[ "Off-by-one l=0", "Not precomputing once" ],
     "quiz":[
       {"q":"Range sum via prefix:", "options":["pref[l]-pref[r]","pref[r]-pref[l-1]","pref[r+1]","arr[r]-arr[l]"], "answer":1, "explain":"Standard formula."},
       {"q":"l=0 case:", "options":["Use pref[-1]","Use pref[r]","Undefined","Add again"], "answer":1, "explain":"Sum(0..r)=pref[r]."}
     ]
    },
    {
     "key":"string","title":"String",
     "what":"Sequence of characters; often immutable.",
     "why":"Text processing everywhere.",
     "analogy":"Beads on a thread.",
     "how":[ "Index/slice like arrays.","Search/count/reverse/compare.","Careful with Unicode & immutability." ],
     "example":"Check if sentence has all vowels.",
     "pitfalls":[ "Off-by-one slices", "Modifying strings per char in loops" ],
     "quiz":[
       {"q":"Strings are often:", "options":["Mutable","Immutable","Graphs","Heaps"], "answer":1, "explain":"New copy on change."},
       {"q":"Index starts at:", "options":["1","0","OS dependent","Random"], "answer":1, "explain":"Usually zero-based."}
     ]
    },
    {
     "key":"recursion","title":"Recursion",
     "what":"Function calls itself to solve smaller instances.",
     "why":"Fits trees, divide-and-conquer, backtracking.",
     "analogy":"Nesting dolls.",
     "how":[ "Base case to stop.","Recursive step reduces input.","Ensure progress." ],
     "example":"Factorial: n! = n × (n−1)!; 0!=1.",
     "pitfalls":[ "Missing base case", "Recomputing overlapping subproblems" ],
     "quiz":[
       {"q":"Must-have in recursion:", "options":["Greedy","Base case","Heap","Queue"], "answer":1, "explain":"Stop condition."},
       {"q":"Often recursive:", "options":["Array access","Tree traversal","Hash lookup","Deque push"], "answer":1, "explain":"Tree shape fits recursion."}
     ]
    },
    {
     "key":"matrix","title":"Matrix/Grid",
     "what":"2D array (rows × cols).",
     "why":"Images, maps, DP tables.",
     "analogy":"Chessboard coordinates.",
     "how":[ "Access a[r][c].","Traverse with boundary checks.","4/8-direction neighbors." ],
     "example":"Count islands by DFS/BFS.",
     "pitfalls":[ "Row/col swap", "Index out of range" ],
     "quiz":[
       {"q":"a[r][c] means:", "options":["a[c][r]","row r, col c","a[r+c]","a[r*c]"], "answer":1, "explain":"Row then column."},
       {"q":"Shortest path in unweighted grid:", "options":["DFS","BFS","Dijkstra","Prim"], "answer":1, "explain":"BFS by layers."}
     ]
    },
    {
     "key":"ll","title":"Linked List",
     "what":"Nodes with data + pointer(s); not contiguous.",
     "why":"Fast insert/delete with node pointer; flexible size.",
     "analogy":"Treasure map: each clue points next.",
     "how":[ "Singly: next.","Doubly: prev & next.","Circular: tail→head." ],
     "example":"Music playlist edits.",
     "pitfalls":[ "No O(1) random access", "Losing head/tail pointers" ],
     "quiz":[
       {"q":"Random access is:", "options":["O(1)","O(log n)","O(n)","O(n log n)"], "answer":2, "explain":"Must traverse."},
       {"q":"DLL has:", "options":["Only next","Only prev","prev & next","No pointers"], "answer":2, "explain":"Both directions."}
     ]
    },
    {
     "key":"stack","title":"Stack",
     "what":"LIFO structure.",
     "why":"Calls, undo/redo, parsing.",
     "analogy":"Plate stack.",
     "how":[ "push/pop/peek top.","Parentheses matching, backtracking." ],
     "example":"Browser back history.",
     "pitfalls":[ "Underflow on empty pop" ],
     "quiz":[
       {"q":"Order:", "options":["FIFO","LIFO","Random","Sorted"], "answer":1, "explain":"Last-in first-out."},
       {"q":"Parentheses check uses:", "options":["Queue","Stack","Heap","Graph"], "answer":1, "explain":"Push opens; pop closes."}
     ]
    },
    {
     "key":"queue","title":"Queue",
     "what":"FIFO structure.",
     "why":"Scheduling, buffering, BFS.",
     "analogy":"Line at ticket counter.",
     "how":[ "enqueue at rear; dequeue at front.","Circular buffers avoid shifting." ],
     "example":"Printer job queue.",
     "pitfalls":[ "Using list.pop(0) repeatedly" ],
     "quiz":[
       {"q":"Order:", "options":["FIFO","LIFO","Random","Sorted"], "answer":0, "explain":"First in, first out."},
       {"q":"BFS uses:", "options":["Stack","Queue","Heap","Set"], "answer":1, "explain":"Level-by-level."}
     ]
    },
    {
     "key":"deque","title":"Deque",
     "what":"Double-ended queue.",
     "why":"Efficient window tricks; palindromes.",
     "analogy":"Two doors entrance/exit.",
     "how":[ "push/pop both ends.","Often O(1) ends." ],
     "example":"Window maximum via monotonic deque.",
     "pitfalls":[ "Treating like list random access" ],
     "quiz":[
       {"q":"Supports:", "options":["Front only","Back only","Both ends","Random O(1)"], "answer":2, "explain":"Double ended."},
       {"q":"Window max uses:", "options":["Monotonic deque","Stack","Heap","Graph"], "answer":0, "explain":"Keep decreasing candidates."}
     ]
    },
    {
     "key":"tree","title":"Tree",
     "what":"Acyclic hierarchical structure.",
     "why":"Filesystems, XML/JSON, indexes.",
     "analogy":"Family tree.",
     "how":[ "Traversals: pre/in/post/level.","Height/depth; balanced vs skewed." ],
     "example":"Filesystem navigation.",
     "pitfalls":[ "Mixing DFS/BFS orders" ],
     "quiz":[
       {"q":"Tree has cycles?", "options":["Yes","No"], "answer":1, "explain":"By definition none."},
       {"q":"Level-order uses:", "options":["Stack","Queue","Heap","Set"], "answer":1, "explain":"BFS with queue."}
     ]
    },
    {
     "key":"heap","title":"Heap",
     "what":"Complete binary tree with heap property.",
     "why":"Priority queue; quick min/max.",
     "analogy":"Smallest box kept on top.",
     "how":[ "Insert bubble-up; remove bubble-down.","Array index representation." ],
     "example":"Task scheduling by priority.",
     "pitfalls":[ "Assuming fully sorted globally" ],
     "quiz":[
       {"q":"Min-heap ensures:", "options":["Every level sorted","Root is smallest","Leaves largest","BST"], "answer":1, "explain":"Only top guaranteed."},
       {"q":"Build-heap cost:", "options":["O(n)","O(n log n)","O(log n)","O(n²)"], "answer":0, "explain":"Bottom-up heapify."}
     ]
    },
    {
     "key":"graph","title":"Graph",
     "what":"Vertices + edges; may have cycles.",
     "why":"Social networks, maps, dependencies.",
     "analogy":"Cities connected by roads.",
     "how":[ "Adjacency list/matrix.","Traverse BFS/DFS; shortest paths Dijkstra/BFS." ],
     "example":"Shortest hops among people.",
     "pitfalls":[ "No visited set → infinite loops" ],
     "quiz":[
       {"q":"Sparse graph structure:", "options":["Matrix","List","Heap","Stack"], "answer":1, "explain":"Adj list scales better."},
       {"q":"Toposort works on:", "options":["Any graph","DAG","Trees only","Bipartite only"], "answer":1, "explain":"Directed acyclic graph."}
     ]
    },
    {
     "key":"greedy","title":"Greedy Algorithm",
     "what":"Pick local best hoping for global optimum.",
     "why":"Fast when property holds.",
     "analogy":"Largest coin first (in canonical systems).",
     "how":[ "Prove greedy-choice & optimal substructure.","Counterexamples show limits." ],
     "example":"Activity selection by earliest finish.",
     "pitfalls":[ "Applying without proof" ],
     "quiz":[
       {"q":"Greedy guaranteed when:", "options":["Feels right","Property proved","Fastest","Two samples"], "answer":1, "explain":"Need proof or known theorem."},
       {"q":"Classic greedy:", "options":["Merge sort","Binary search","Activity selection","Matrix-chain DP"], "answer":2, "explain":"Pick compatible earliest finish."}
     ]
    },
]

# -------------------------------------------------------
# Sidebar: Navigation (Pages)
# -------------------------------------------------------
st.sidebar.title("🎓 Data Structure-AI- Tutor")
st.sidebar.caption("DSA Concept Coach — No-Code Learning Mode")

PAGE_OPTIONS = ["Learn (No-Code)", "Practice (Topic-wise 10Q)"]
page = st.sidebar.radio("Go to", PAGE_OPTIONS, index=0)

# Confidence tracker (for Learn page)
if "confidence" not in st.session_state:
    st.session_state["confidence"] = {t["key"]: None for t in topics}

# -------------------------------------------------------
# PAGE: Learn (No-Code)
# -------------------------------------------------------
# if page == "Learn (No-Code)":
#     topic_names = [f"{i+1}. {t['title']}" for i,t in enumerate(topics)]
#     choice = st.sidebar.radio("Jump to a topic", topic_names, index=0)
#     idx = topic_names.index(choice)
#     topic = topics[idx]

#     st.sidebar.markdown("---")
#     cur_conf = st.sidebar.slider(
#         "Your confidence in this topic (0–100)", 0, 100,
#         value=st.session_state["confidence"].get(topic["key"]) or 50
#     )
#     if st.sidebar.button("Save confidence"):
#         st.session_state["confidence"][topic["key"]] = cur_conf
#         st.sidebar.success("Saved!")

#     vals = [v for v in st.session_state["confidence"].values() if v is not None]
#     if vals:
#         avg = sum(vals)/len(vals)
#         st.sidebar.metric("Overall confidence", f"{avg:.0f}%")

#     st.sidebar.markdown("---")
#     st.sidebar.write("Tips")
#     st.sidebar.write("• Read the **What/Why/How** first.")
#     st.sidebar.write("• Study the **example & pitfalls**.")
#     st.sidebar.write("• Try the **quiz** for instant feedback.")

#     # ---------- Main content ----------
#     st.title(choice)
#     chip("No-Code Learning"); chip("Visual Intuition"); chip("Beginner-Friendly")

#     col1, col2 = st.columns([1.2, 1])
#     with col1:
#         titled_box("What it means", topic["what"])
#         titled_box("Why it matters", topic["why"])
#         titled_box("Analogy", topic.get("analogy",""))

#         st.markdown("### How to think about it")
#         for step in topic["how"]:
#             st.markdown(f"- {step}")

#         st.markdown("### A tiny, concrete example")
#         st.write(topic["example"])

#         st.markdown("### Common pitfalls")
#         for p in topic["pitfalls"]:
#             st.markdown(f"- {p}")
    
#         # ---------- Live, no-code demo ----------
#     # st.markdown("### ▶ Run a live demo")
#     # demo = DEMOS.get(topic["key"])
#     # if not demo:
#     #     st.info("Demo coming soon for this topic.")
#     # else:
#     #     _show_code(demo.code)
#     #     if st.button(f"Run: {demo.title}", key=f"learn_run_{idx}_{topic['key']}"):

#     #         with st.spinner("Running demo..."):
#     #             try:
#     #                 demo.runner()
#     #                 st.success("Done.")
#     #                 card("tip", "How it works", demo.explainer)
#     #             except Exception as e:
#     #                 card("warn", "Demo error", f"{e}")

#     # st.markdown("---")
#     # quiz_block(topic["quiz"], key_prefix=f"mini_{topic['key']}")

#         # ---------- Live, no-code demo ----------
#     st.markdown("### ▶ Run a live demo")
#     demo = DEMOS.get(topic["key"])
#     if not demo:
#         st.info("Demo coming soon for this topic.")
#     else:
#         _show_code(demo.code)
#         if st.button(f"Run: {demo.title}", key=f"run_{topic['key']}"):
#             with st.spinner("Running demo..."):
#                 try:
#                     demo.runner()
#                     st.success("Done.")
#                     card("tip", "How it works", demo.explainer)
#                 except Exception as e:
#                     card("warn", "Demo error", f"{e}")

#     st.markdown("---")
#     quiz_block(topic["quiz"], key_prefix=f"mini_{topic['key']}_{idx}")



#     with col2:
#         st.markdown("### Visual intuition")
#         key = topic["key"]
#         try:
#             if key == "array":
#                 fig, ax = start_canvas((9,2.4)); draw_array(ax, [7,2,9,3,1], title="Array with indices")
#             elif key == "search":
#                 fig, ax = start_canvas((9,2.4)); draw_array(ax, [3,6,9,12,15,22,30], title="Search target within array")
#             elif key == "sorting":
#                 fig, ax = start_canvas((9,2.4)); draw_array(ax, sorted([7,2,9,3,1]), title="After Sorting")
#             elif key == "hashing":
#                 fig, ax = start_canvas((7,4))
#                 # simple buckets demo
#                 for i in range(5):
#                     ax.add_patch(Rectangle((0, -i), 3.5, 1, fill=False, linewidth=2))
#                     ax.text(-0.4, -i+0.5, f"{i}", ha="right", va="center", fontsize=11, color="#64748b")
#                 for k in ["amy","bob","zoe","ann","rob"]:
#                     hv = sum(ord(c) for c in k) % 5
#                     ax.text(0.2 + np.random.rand()*3.0, -hv+0.5 + (np.random.rand()-0.5)*0.4, k, fontsize=11)
#                 ax.set_xlim(-1.0, 4.2); ax.set_ylim(-5.5, 1.0); ax.axis("off"); ax.set_title("Buckets (index = hash(key) % m)", fontsize=13, pad=6)
#             elif key == "twoptr":
#                 fig, ax = start_canvas((9,2.4)); draw_two_pointers(ax, [1,2,3,7,8,12], 0, 5, "Start i=0, j=n-1")
#             elif key == "window":
#                 fig, ax = start_canvas((9,2.4)); draw_window(ax, [4,2,1,7,8,1,2,8,1,0], 0, 2, "Window size 3")
#             elif key == "prefix":
#                 fig, ax = start_canvas((9,4)); draw_prefix(ax, [2,1,3,4,2,1], "Prefix sums")
#             elif key == "string":
#                 fig, ax = start_canvas((10,2.4)); draw_chars(ax, "hello world", 4, "Characters by index")
#             elif key == "recursion":
#                 fig, ax = start_canvas((5,3.6)); draw_stack(ax, ["fact(4)","fact(3)","fact(2)","fact(1)","fact(0)"], "Call Stack (top at right)")
#             elif key == "matrix":
#                 fig, ax = start_canvas((6,2.4)); draw_array(ax, ["r0c0","r0c1","r0c2","r1c0","r1c1","r1c2"], title="Flattened view (concept)")
#             elif key == "ll":
#                 fig, ax = start_canvas((10,2.4)); draw_array(ax, ["[10|•]","[20|•]","[30|•]"], title="Linked nodes concept")
#             elif key == "stack":
#                 fig, ax = start_canvas((5,3.6)); draw_stack(ax, ["A","B","C"], "Top at right")
#             elif key == "queue":
#                 fig, ax = start_canvas((8,2.4)); draw_queue(ax, ["10","20","30","40"], "Front → ... → Rear")
#             elif key == "deque":
#                 fig, ax = start_canvas((8,2.4)); draw_deque(ax, ["L","M","N","O"], "Deque ends")
#             elif key == "tree":
#                 fig, ax = start_canvas((8,4.2)); draw_tree(ax, example_tree(), "Binary Tree (concept)")
#             elif key == "heap":
#                 fig, ax = start_canvas((7,4)); draw_heap(ax, [1,3,5,7,9,6,8], "Min-Heap property")
#             elif key == "graph":
#                 fig, ax = start_canvas((6.5,4)); draw_graph(ax, [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)], "Small graph")
#             elif key == "greedy":
#                 fig, ax = start_canvas((8,2.4)); draw_array(ax, ["25","10","5","1"], title="Pick largest coin first (canonical systems)")
#             else:
#                 fig = None
#             if fig:
#                 # st.pyplot(fig, use_container_width=True)
#                 st.pyplot(fig, width="stretch")

#             else:
#                 st.info("This topic is conceptual — read the panels on the left.")
#         except Exception as e:
#             st.warning(f"Visual not available: {e}")

#     st.markdown("---")
#     quiz_block(topic["quiz"], key_prefix=f"mini_{topic['key']}")
#     st.markdown("---")
#     st.caption("© Professor CoderBuddy AI — Learn by intuition, examples, and quick quizzes.")

# # -------------------------------------------------------
# # PAGE: Practice (Topic-wise 10Q)
# # -------------------------------------------------------
# elif page == "Practice (Topic-wise 10Q)":
#     st.title("Practice (Topic-wise) — 10 Questions Each")
#     badge("Per-topic quizzes"); badge("Score tracking")

#     practice_topics = [{"key": t["key"], "title": t["title"]} for t in topics]

#     try:
#         topic_names = [f"{i+1}. {t['title']}" for i, t in enumerate(practice_topics)]
#         cur_name = st.selectbox("Choose a topic to practice", topic_names, index=0)
#         cur_idx = topic_names.index(cur_name)
#         cur_topic = practice_topics[cur_idx]

#         render_topic_practice(cur_topic["key"], store_key=f"score_{cur_topic['key']}")
#     except Exception as e:
#         card("warn", "Topic list unavailable", f"Could not load topics. Error: {e}")
#         st.stop()

#     st.markdown("---")
#     st.subheader("Your Progress")

#     scores = st.session_state.get("scores", {})
#     attempted = []
#     for t in practice_topics:
#         key = f"score_{t['key']}"
#         if key in scores:
#             attempted.append((t["title"], scores[key]))

#     if not attempted:
#         card("note", "No attempts yet", "Take a topic quiz above to see your progress here.")
#     else:
#         for title, sc in attempted:
#             pct = int(100 * sc / 10)
#             st.write(f"**{title}** — {sc}/10")
#             st.progress(pct / 100)

#         avg = sum(sc for _, sc in attempted) / (10 * len(attempted))
#         st.markdown("### Overall")
#         st.progress(avg)
#         st.caption(f"Average across attempted topics: {avg*100:.0f}%")



# -------------------------------------------------------
# PAGE: Practice (Topic-wise 10Q)
# -------------------------------------------------------
elif page == "Practice (Topic-wise 10Q)":
    st.title("Practice (Topic-wise) — 10 Questions Each")
    badge("Per-topic quizzes"); badge("Score tracking")

    # Build the list from the 'topics' meta (keys must match TOPIC_BANKS keys)
    practice_topics = [{"key": t["key"], "title": t["title"]} for t in topics]

    # --- Topic picker (unique key) ---
    topic_names = [f"{i+1}. {t['title']}" for i, t in enumerate(practice_topics)]
    cur_name = st.selectbox("Choose a topic to practice", topic_names, index=0, key="practice_topic_select")
    cur_idx = topic_names.index(cur_name)
    cur_topic = practice_topics[cur_idx]
    cur_key = cur_topic["key"]

    # --- Load questions and render the bank ---
    qs = get_topic_bank(cur_key)

    # (Temporary) tiny debug line so you can verify loading is OK; remove later.
    st.caption(f"Loaded topic key: **{cur_key}** · Questions found: **{len(qs)}**")

    if not qs:
        # If this triggers, your key in TOPIC_BANKS is missing/mismatched.
        card("warn", "No questions found",
             f"No question bank mapped for key '{cur_key}'. "
             "Check TOPIC_BANKS and the bank_...() function names.")
    else:
        st.subheader("Practice: 10 Questions")
        # Key prefix includes the topic key so radios are unique & collision-free.
        quiz_block_with_score(qs, key_prefix=f"bank_{cur_key}", store_key=f"score_{cur_key}")

    st.markdown("---")
    st.subheader("Your Progress")

    scores = st.session_state.get("scores", {})
    attempted = []
    for t in practice_topics:
        k = f"score_{t['key']}"
        if k in scores:
            attempted.append((t["title"], scores[k]))

    if not attempted:
        card("note", "No attempts yet", "Take a topic quiz above to see your progress here.")
    else:
        for title, sc in attempted:
            pct = int(100 * sc / 10)
            st.write(f"**{title}** — {sc}/10")
            st.progress(pct / 100)

        avg = sum(sc for _, sc in attempted) / (10 * len(attempted))
        st.markdown("### Overall")
        st.progress(avg)
        st.caption(f"Average across attempted topics: {avg*100:.0f}%")

# -------------------------------------------------------
# PAGE: Learn (No-Code)
# -------------------------------------------------------
if page == "Learn (No-Code)":
    topic_names = [f"{i+1}. {t['title']}" for i, t in enumerate(topics)]
    choice = st.sidebar.radio("Jump to a topic", topic_names, index=0)
    idx = topic_names.index(choice)
    topic = topics[idx]

    st.sidebar.markdown("---")
    cur_conf = st.sidebar.slider(
        "Your confidence in this topic (0–100)", 0, 100,
        value=st.session_state["confidence"].get(topic["key"]) or 50
    )
    if st.sidebar.button("Save confidence"):
        st.session_state["confidence"][topic["key"]] = cur_conf
        st.sidebar.success("Saved!")

    vals = [v for v in st.session_state["confidence"].values() if v is not None]
    if vals:
        avg = sum(vals) / len(vals)
        st.sidebar.metric("Overall confidence", f"{avg:.0f}%")

    st.sidebar.markdown("---")
    st.sidebar.write("Tips")
    st.sidebar.write("• Read the **What/Why/How** first.")
    st.sidebar.write("• Study the **example & pitfalls**.")
    st.sidebar.write("• Try the **quiz** for instant feedback.")

    # ---------- Main content ----------
    st.title(choice)
    chip("No-Code Learning"); chip("Visual Intuition"); chip("Beginner-Friendly")

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        titled_box("What it means", topic["what"])
        titled_box("Why it matters", topic["why"])
        titled_box("Analogy", topic.get("analogy", ""))

        st.markdown("### How to think about it")
        for step in topic["how"]:
            st.markdown(f"- {step}")

        st.markdown("### A tiny, concrete example")
        st.write(topic["example"])

        st.markdown("### Common pitfalls")
        for p in topic["pitfalls"]:
            st.markdown(f"- {p}")

    with col2:
        st.markdown("### Visual intuition")
        key = topic["key"]
        try:
            if key == "array":
                fig, ax = start_canvas((9, 2.4)); draw_array(ax, [7, 2, 9, 3, 1], title="Array with indices")
            elif key == "search":
                fig, ax = start_canvas((9, 2.4)); draw_array(ax, [3, 6, 9, 12, 15, 22, 30], title="Search target within array")
            elif key == "sorting":
                fig, ax = start_canvas((9, 2.4)); draw_array(ax, sorted([7, 2, 9, 3, 1]), title="After Sorting")
            elif key == "hashing":
                fig, ax = start_canvas((7, 4))
                for i in range(5):
                    ax.add_patch(Rectangle((0, -i), 3.5, 1, fill=False, linewidth=2))
                    ax.text(-0.4, -i + 0.5, f"{i}", ha="right", va="center", fontsize=11, color="#64748b")
                for k in ["amy", "bob", "zoe", "ann", "rob"]:
                    hv = sum(ord(c) for c in k) % 5
                    ax.text(0.2 + np.random.rand() * 3.0, -hv + 0.5 + (np.random.rand() - 0.5) * 0.4, k, fontsize=11)
                ax.set_xlim(-1.0, 4.2); ax.set_ylim(-5.5, 1.0); ax.axis("off"); ax.set_title("Buckets (index = hash(key) % m)", fontsize=13, pad=6)
            elif key == "twoptr":
                fig, ax = start_canvas((9, 2.4)); draw_two_pointers(ax, [1, 2, 3, 7, 8, 12], 0, 5, "Start i=0, j=n-1")
            elif key == "window":
                fig, ax = start_canvas((9, 2.4)); draw_window(ax, [4, 2, 1, 7, 8, 1, 2, 8, 1, 0], 0, 2, "Window size 3")
            elif key == "prefix":
                fig, ax = start_canvas((9, 4)); draw_prefix(ax, [2, 1, 3, 4, 2, 1], "Prefix sums")
            elif key == "string":
                fig, ax = start_canvas((10, 2.4)); draw_chars(ax, "hello world", 4, "Characters by index")
            elif key == "recursion":
                fig, ax = start_canvas((5, 3.6)); draw_stack(ax, ["fact(4)", "fact(3)", "fact(2)", "fact(1)", "fact(0)"], "Call Stack (top at right)")
            elif key == "matrix":
                fig, ax = start_canvas((6, 2.4)); draw_array(ax, ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"], title="Flattened view (concept)")
            elif key == "ll":
                fig, ax = start_canvas((10, 2.4)); draw_array(ax, ["[10|•]", "[20|•]", "[30|•]"], title="Linked nodes concept")
            elif key == "stack":
                fig, ax = start_canvas((5, 3.6)); draw_stack(ax, ["A", "B", "C"], "Top at right")
            elif key == "queue":
                fig, ax = start_canvas((8, 2.4)); draw_queue(ax, ["10", "20", "30", "40"], "Front → ... → Rear")
            elif key == "deque":
                fig, ax = start_canvas((8, 2.4)); draw_deque(ax, ["L", "M", "N", "O"], "Deque ends")
            elif key == "tree":
                fig, ax = start_canvas((8, 4.2)); draw_tree(ax, example_tree(), "Binary Tree (concept)")
            elif key == "heap":
                fig, ax = start_canvas((7, 4)); draw_heap(ax, [1, 3, 5, 7, 9, 6, 8], "Min-Heap property")
            elif key == "graph":
                fig, ax = start_canvas((6.5, 4)); draw_graph(ax, [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)], "Small graph")
            elif key == "greedy":
                fig, ax = start_canvas((8, 2.4)); draw_array(ax, ["25", "10", "5", "1"], title="Pick largest coin first (canonical systems)")
            else:
                fig = None

            if fig:
                st.pyplot(fig, width="stretch")
            else:
                st.info("This topic is conceptual — read the panels on the left.")
        except Exception as e:
            st.warning(f"Visual not available: {e}")

    # ---------- Live, no-code demo (one copy only) ----------
    st.markdown("### ▶ Run a live demo")
    demo = DEMOS.get(topic["key"])
    if not demo:
        st.info("Demo coming soon for this topic.")
    else:
        _show_code(demo.code)
        if st.button(f"Run: {demo.title}", key=f"learn_run_{idx}_{topic['key']}"):
            with st.spinner("Running demo..."):
                try:
                    demo.runner()
                    st.success("Done.")
                    card("tip", "How it works", demo.explainer)
                except Exception as e:
                    card("warn", "Demo error", f"{e}")

    st.markdown("---")
    # Mini quiz (one copy only) — note unique key_prefix
    quiz_block(topic["quiz"], key_prefix=f"mini_{topic['key']}_{idx}")
    st.markdown("---")
    st.caption(" Data Structure-AI- Tutor — Learn by intuition, examples, and quick quizzes.")
