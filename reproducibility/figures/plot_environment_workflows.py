#!/usr/bin/env python3
"""Exact DRAMA layouts and per-cell precedence graphs, not a result illustration.

Easy/Hard/Large coordinates: active manuscript Section 5.1.1.
Cross: frozen workflow tree src/sacc-drama/DRAMA/DRAMA/specs.py,
SHA256 eec332ba86a990b362a56befff5e46c56240dff41b06b45b198edf08b15bd1fc.
Retrieved read-only 2026-09-28. Coordinates are zero-based (x,y), y upward.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_plot_style import (configure_style, assert_artists_inside, save_figure,
                              BLUE, ORANGE, GOLD, RED, INK, GRID, PANEL, POINT)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parent
LAYOUTS = [
    ('Easy',6,4,[(2,2),(2,3)],[]),
    ('Hard',6,4,[(0,1),(0,2),(1,2),(0,4),(1,4),(1,5),(4,0),(5,0),(4,4),(5,5)],[(3,3)]),
    ('Large',8,8,[(x,y) for xs,ys in [([1,2,3],[2,3]),([1,2,3],[4,5]),([4,5,6],[1,2]),([4,5,6],[5,6])] for x in xs for y in ys],[(x,y) for x in [4,5,6] for y in [3,4]]),
    ('Cross-5',7,5,[(1,2),(1,3),(2,2),(2,3),(3,1),(4,1),(3,2),(4,2),(4,4),(4,5),(5,4),(5,5)],[(3,3),(3,4),(5,2),(5,3)]),
]

def main():
    configure_style()
    fig = plt.figure(figsize=(7,4.55))
    checked=[]
    for i,(name,n,robots,cells,obstacles) in enumerate(LAYOUTS):
        assert len(cells)==len(set(cells)) and not set(cells)&set(obstacles)
        assert all(0<=x<n and 0<=y<n for x,y in cells+obstacles)
        ax=fig.add_axes([.035+i*.242,.535,.212,.405])
        ax.set_xlim(-.7,n+.2); ax.set_ylim(-1.3,n+2); ax.set_aspect('equal'); ax.axis('off')
        for x in range(n):
            for y in range(n):
                target=(x,y) in cells; blocked=(x,y) in obstacles
                ax.add_patch(Rectangle((x,y),1,1,facecolor=INK if blocked else BLUE if target else PANEL,
                                       edgecolor=GRID,linewidth=.5,hatch='///' if blocked else None))
        title=ax.text((n-.1)/2,n+1.23,f'({chr(97+i)}) {name}',ha='center',weight='semibold',fontsize=9)
        sub=ax.text((n-.1)/2,n+.55,f'{robots} robots | {len(cells)} cells',ha='center',fontsize=7)
        bottom=ax.text((n-.1)/2,-1.04,f'{n} x {n} grid | {len(obstacles)} obstacles',ha='center',fontsize=6.9)
        for v in [0,n-1]:
            ax.text(v+.5,-.35,str(v),fontsize=6.5,ha='center')
            ax.text(-.3,v+.5,str(v),fontsize=6.5,va='center',ha='right')
        checked.append((ax,[title,sub,bottom]))
    legend_ax=fig.add_axes([.05,.50,.9,.06]); legend_ax.axis('off')
    leg=legend_ax.legend(handles=[Patch(facecolor=BLUE,label='Construction cell'),Patch(facecolor=INK,hatch='///',label='Static obstacle'),Patch(facecolor=PANEL,edgecolor=GRID,label='Traversable cell')],
                         loc='center',ncol=3,fontsize=7.3)
    checked.append((legend_ax,[leg]))

    def setup(bounds):
        ax=fig.add_axes(bounds); ax.set_xlim(-.12,10.12); ax.set_ylim(0,3); ax.axis('off'); return ax
    def box(ax,x,y,w,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,.57,boxstyle='round,pad=0.045,rounding_size=0.07',facecolor=PANEL,edgecolor=color,linewidth=1.3))
        return ax.text(x+w/2,y+.285,text,ha='center',va='center',fontsize=7.5,color=INK)
    def arrow(ax,a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,color=INK,linewidth=1.1))
    ax=setup([.06,.315,.89,.175])
    texts=[ax.text(0,2.55,'(e) Easy / Hard / Large: four stages per cell',fontsize=8.6,weight='semibold')]
    for x,label,color in [(0,'Distribute',BLUE),(2.6,'Vibrate',ORANGE),(5.2,'Level',GOLD),(7.8,'Cover',RED)]:
        texts.append(box(ax,x,1.30,2.0,label,color))
        if x<7.8: arrow(ax,(x+2.06,1.585),(x+2.50,1.585))
    texts.append(ax.text(0,.42,'Large scales the map, team and task count; the precedence chain stays the same.',fontsize=7.2))
    checked.append((ax,texts))

    ax=setup([.06,.025,.89,.285])
    texts=[ax.text(0,2.69,'(f) Cross-5: parallel branches, AND join and role re-entry',fontsize=8.6,weight='semibold')]
    nodes={'1a':(0,1.2,BLUE),'2a':(2,1.87,ORANGE),'3':(2,.54,GOLD),'1b':(4,1.2,BLUE),
           '2b':(5.7,1.2,ORANGE),'4':(7.4,1.2,RED),'5':(9,1.2,POINT)}
    width=.87
    for node,(x,y,c) in nodes.items(): texts.append(box(ax,x,y,width,node,c))
    edges=[('1a','2a'),('1a','3'),('2a','1b'),('3','1b'),('1b','2b'),('2b','4'),('4','5')]
    for u,v in edges:
        x,y,_=nodes[u]; xx,yy,_=nodes[v]
        arrow(ax,(x+width+.07,y+.285),(xx-.08,yy+.285))
    texts.append(ax.text(4.43,.83,'AND',ha='center',fontsize=7.2,weight='semibold'))
    texts.append(ax.text(0,.13,'Node number = role; a/b = first/second visit. Both branches release 1b only after completion.',fontsize=7.1))
    checked.append((ax,texts))
    for ax,artists in checked: assert_artists_inside(fig,ax,artists,tolerance=2)
    for path in save_figure(fig,ROOT/'environment_workflow_structure'): print(path)

if __name__=='__main__': main()
