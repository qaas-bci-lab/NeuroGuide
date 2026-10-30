"""Readable 72-channel 10-10 schematic linked to individual placed coordinates."""
import csv
from PyQt6.QtWidgets import QDialog,QVBoxLayout,QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import Circle,Ellipse
import numpy as np

ROWS=[['Fp1','Fpz','Fp2'],['AF7','AF3','AFz','AF4','AF8'],
      ['F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10'],
      ['FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10'],
      ['T7','C5','C3','C1','Cz','C2','C4','C6','T8'],
      ['TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8'],
      ['P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10'],
      ['PO9','PO7','PO3','POz','PO4','PO8','PO10'],['O9','O1','Oz','O2','O10'],['Iz']]


def electrode_dialog(owner,path,result=None):
    dialog=QDialog(owner);dialog.setWindowTitle('个体 10-10 电极方案');dialog.resize(780,840)
    layout=QVBoxLayout(dialog)
    title=QLabel('10-10 电极位置 · 头顶视图');title.setStyleSheet('font-size:18px;font-weight:bold;');layout.addWidget(title)
    hint=QLabel('上方为额头，左侧为左耳。点击电极可查看个体 MRI 中的放置坐标。');hint.setWordWrap(True);layout.addWidget(hint)
    with open(path,encoding='utf-8',newline='') as stream:records=list(csv.DictReader(stream,delimiter='\t'))
    current={}
    from ui.result_text import ANODE_COLOR,CATHODE_COLOR,electrode_scheme_html
    palette=[ANODE_COLOR,CATHODE_COLOR]
    for channel,(pair,value) in enumerate(zip((result or {}).get('pairs',[]),(result or {}).get('currents_peak_ma',[]))):
        current[pair[0]]=(channel,float(value));current[pair[1]]=(channel,-float(value))
    tes=(result or {}).get('stimulation_type')=='tES'
    if tes:
        current={int(label):(0,float(value)) for label,value in zip(result['electrode_labels'],result['currents_ma'])}
    fig=Figure(facecolor='#181b22');canvas=FigureCanvasQTAgg(fig);layout.addWidget(canvas,1)
    ax=fig.add_subplot(111);fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.03)
    ax.set_aspect('equal');ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.13,1.2);ax.axis('off')
    ax.add_patch(Circle((0,0),1,fill=False,edgecolor='#b2bdcf',linewidth=2))
    ax.plot([-.16,0,.16], [.99,1.15,.99],color='#b2bdcf',linewidth=2)
    for x in (-1,1):ax.add_patch(Ellipse((x,0),.16,.45,fill=False,edgecolor='#b2bdcf',linewidth=2))
    positions={};dots=[];lookup={row['name']:(index,row) for index,row in enumerate(records,1)}
    for row_index,row in enumerate(ROWS):
        y=.78-row_index*.185 if row_index<9 else -1.
        extent=min(.94,np.sqrt(max(0,1-y*y))*.98)
        xs=np.linspace(-extent,extent,len(row)) if len(row)>1 else [0]
        if row_index==0:xs=[-.26,0,.26]
        if row_index==1:xs=np.linspace(-.61,.61,5)
        for x,name in zip(xs,row):
            if name not in lookup:continue
            index,record=lookup[name];positions[name]=(float(x),float(y))
            selected=current.get(index);color=palette[0 if selected[1]>0 else 1] if selected else '#323b4d'
            marker = 's' if selected and not tes and selected[0] == 1 else 'o'
            dot=ax.scatter([x],[y],s=690,c=color,marker=marker,edgecolors='#ecf2ff' if selected else '#56637b',linewidths=2 if selected else .7,picker=8,zorder=3)
            dot.electrode_record=(index,record);dots.append(dot)
            ax.text(x,y,name,color='white' if selected else '#e0e7f2',fontsize=10,ha='center',va='center',zorder=4)
            if selected:
                tag = '' if tes else f'{selected[0]+1}{"+" if selected[1]>0 else "-"} · '
                ax.text(x,y-.075,tag+f'{selected[1]:+g}',color='white',fontsize=11,fontweight='bold',ha='center',va='top',zorder=5,
                    bbox={'boxstyle':'round,pad=.18','facecolor':color,'edgecolor':'none'})
    names={index:row['name'] for index,row in enumerate(records,1)}
    pair_lines=[]
    if not tes:
        for channel,pair in enumerate((result or {}).get('pairs',[])):
            endpoints=[positions.get(names.get(label)) for label in pair]
            if len(endpoints)==2 and all(point is not None for point in endpoints):
                line,=ax.plot([point[0] for point in endpoints],[point[1] for point in endpoints],
                              color='#aab6c9',linewidth=1.7,linestyle='-' if channel==0 else '--',zorder=2)
                pair_lines.append(line)
    legend=QLabel(electrode_scheme_html(result,names) if result else '全部 72 电极已放置；计算方案后会高亮所用电极。')
    legend.setWordWrap(True);layout.addWidget(legend)
    if result and not tes:
        legend.setText('第1对：圆形、实线、1+/1−；第2对：方形、虚线、2+/2−。<br>'+legend.text())
    details=QLabel('深红为阳极，深蓝为阴极；数字为峰值电流（mA）。此图为位置示意。');details.setWordWrap(True);layout.addWidget(details)
    if tes:details.setText('深红为正向，深蓝为负向；数字为直流电流或交流峰值（mA）。交流按同频 0°/180°相位计算。')
    elif result:details.setText('深红为阳极，深蓝为阴极；电流单位 mA。连线仅表示电极配对，不表示脑内电流路径。')
    def clicked(event):
        index,record=event.artist.electrode_record
        coordinates='，'.join(f'{float(record[key]):.1f}' for key in ('ras_x','ras_y','ras_z'))
        chosen=current.get(index)
        details.setText(f"{record['name']} · 个体 RAS ({coordinates}) mm"+ ((f' · tES {chosen[1]:+g} mA' if tes else f' · 通道{chosen[0]+1} {chosen[1]:+g} mA 峰值') if chosen else ' · 未施加电流'))
    canvas.mpl_connect('pick_event',clicked)
    dialog.electrode_positions=positions;dialog.electrode_currents=current;dialog.figure_canvas=canvas
    dialog.electrode_pair_lines=pair_lines;dialog.scheme_legend=legend
    return dialog
